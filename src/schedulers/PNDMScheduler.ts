import { cat, linspace, range } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'

export interface SchedulerConfig {
  'beta_end': number,
  'beta_schedule': string,
  'beta_start': number,
  'clip_sample': boolean,
  'num_train_timesteps': number,
  prediction_type?: 'epsilon'|'v_prediction',
  'set_alpha_to_one': boolean,
  'skip_prk_steps': boolean,
  'steps_offset': number,
  'trained_betas': null
}

/**
 * Pseudo numerical methods for diffusion models (PNDM) proposes using more advanced ODE integration techniques,
 * namely Runge-Kutta method and a linear multistep method.
 */
export class PNDMScheduler {
  betas: Tensor
  alphas: Tensor
  alphasCumprod!: Tensor
  finalAlphaCumprod!: number
  initNoiseSigma: number
  pndmOrder: number
  curModelOutput: Tensor|number
  counter: number
  curSample: Tensor|null
  ets: Tensor[]
  numInferenceSteps: number = 20
  timesteps: Tensor
  prkTimesteps: Tensor|null
  plmsTimesteps: Tensor|null
  config: SchedulerConfig // Define your config type

  constructor (
    config: SchedulerConfig,
  ) {
    this.config = config

    if (config.trained_betas !== null) {
      this.betas = linspace(config.beta_start, config.beta_end, config.num_train_timesteps)
    } else if (config.beta_schedule === 'linear') {
      this.betas = linspace(config.beta_start, config.beta_end, config.num_train_timesteps)
    } else if (config.beta_schedule === 'scaled_linear') {
      this.betas = linspace(config.beta_start ** 0.5, config.beta_end ** 0.5, config.num_train_timesteps).pow(2)
    } else if (config.beta_schedule === 'squaredcos_cap_v2') {
      this.betas = betasForAlphaBar(config.num_train_timesteps)
    } else {
      throw new Error(`${config.beta_schedule} does is not implemented for ${this.constructor}`)
    }

    this.alphas = linspace(1, 1, config.num_train_timesteps).sub(this.betas)

    this.initNoiseSigma = 1.0
    this.pndmOrder = 4

    // running values
    this.curModelOutput = 0
    this.counter = 0
    this.curSample = null
    this.ets = []

    // setable values
    this.timesteps = range(0, config.num_train_timesteps).reverse()
    this.prkTimesteps = null
    this.plmsTimesteps = null

    this.alphasCumprod = this.alphas.cumprod()
    this.finalAlphaCumprod = config.set_alpha_to_one ? 1.0 : this.alphasCumprod[0].data
  }

  setTimesteps (numInferenceSteps: number) {
    this.numInferenceSteps = numInferenceSteps
    const stepRatio = ~~(this.config.num_train_timesteps / this.numInferenceSteps)
    this.timesteps = range(0, numInferenceSteps).mul(stepRatio).round()
    this.timesteps = this.timesteps.add(this.config.steps_offset)

    if (this.config.skip_prk_steps) {
      this.prkTimesteps = new Tensor(new Int32Array())
      const size = this.timesteps.size
      this.plmsTimesteps = cat([
        this.timesteps.slice([0, size - 1]),
        this.timesteps.slice([size - 2, size - 1]),
        this.timesteps.slice([size - 1, size]),
      ]).reverse().clone()
      this.timesteps = this.plmsTimesteps!
    } else {
      const prkTimesteps = this.timesteps.slice(-this.pndmOrder)
        .tile([2])
        .add(
          // tf.tensor([0, this.config.num_train_timesteps / numInferenceSteps / 2]).tile([this.pndmOrder])
        )
      this.prkTimesteps = prkTimesteps.slice(0, -1).tile([2]).slice(1, -1).reverse().clone()
      this.plmsTimesteps = this.timesteps.slice(0, -3).reverse().clone()
      this.timesteps = cat([this.prkTimesteps!, this.plmsTimesteps!])
    }

    this.ets = []
    this.counter = 0
    this.curModelOutput = 0
  }

  step (
    modelOutput: Tensor,
    timestep: number,
    sample: Tensor,
  ) {
    if (!this.config.skip_prk_steps && this.counter < this.prkTimesteps!.dims[0]) {
      return this.stepPrk(modelOutput, timestep, sample)
    } else {
      return this.stepPlms(modelOutput, timestep, sample)
    }
  }

  stepPrk (
    modelOutput: Tensor,
    timestep: number,
    sample: Tensor,
  ) {
    if (this.numInferenceSteps === null) {
      throw new Error(
        "Number of inference steps is 'null', you need to run 'setTimesteps' after creating the scheduler",
      )
    }

    const diffToPrev = this.counter % 2 === 0 ? this.config.num_train_timesteps / this.numInferenceSteps / 2 : 0
    const prevTimestep = timestep - diffToPrev
    timestep = this.prkTimesteps!.get(this.counter / 4 * 4)

    if (this.counter % 4 === 0) {
      this.curModelOutput = (this.curModelOutput as Tensor).add(modelOutput.mul(1 / 6))
      this.ets.push(modelOutput)
      this.curSample = sample
    } else if ((this.counter - 1) % 4 === 0) {
      this.curModelOutput = (this.curModelOutput as Tensor).add(modelOutput.mul(1 / 3))
    } else if ((this.counter - 2) % 4 === 0) {
      this.curModelOutput = (this.curModelOutput as Tensor).add(modelOutput.mul(1 / 3))
    } else if ((this.counter - 3) % 4 === 0) {
      modelOutput = (this.curModelOutput as Tensor).add(modelOutput.mul(1 / 6))
      this.curModelOutput = 0
    }
    const curSample = this.curSample !== null ? this.curSample : sample
    const prevSample = this._getPrevSample(curSample, timestep, prevTimestep, modelOutput)
    this.counter += 1

    return prevSample
  }

  stepPlms (
    modelOutput: Tensor,
    timestep: number,
    sample: Tensor,
  ) {
    let prevTimestep = timestep - ~~(this.config.num_train_timesteps / this.numInferenceSteps)

    if (this.counter !== 1) {
      this.ets = this.ets.slice(-3)
      this.ets.push(modelOutput)
    } else {
      prevTimestep = timestep
      timestep = timestep + ~~(this.config.num_train_timesteps / this.numInferenceSteps)
    }

    if (this.ets.length === 1 && this.counter === 0) {
      this.curSample = sample
    } else if (this.ets.length === 1 && this.counter === 1) {
      modelOutput = modelOutput.add(this.ets[this.ets.length - 1]).div(2)
      sample = this.curSample!
      this.curSample = null
    } else if (this.ets.length === 2) {
      modelOutput = this.ets[this.ets.length - 1].mul(3)
        .sub(this.ets[this.ets.length - 2])
        .div(2)
    } else if (this.ets.length === 3) {
      modelOutput =
        this.ets[this.ets.length - 1].mul(23)
          .sub(
            this.ets[this.ets.length - 2].mul(16),
          )
          .add(
            this.ets[this.ets.length - 3].mul(5),
          )
          .div(12)
    } else {
      modelOutput =
        this.ets[this.ets.length - 1].mul(55)
          .sub(
            this.ets[this.ets.length - 2].mul(59),
          )
          .add(
            this.ets[this.ets.length - 3].mul(37),
          )
          .sub(
            this.ets[this.ets.length - 4].mul(9),
          ).mul(1 / 24)
    }
    const prevSample = this._getPrevSample(sample, timestep, prevTimestep, modelOutput)
    this.counter += 1

    return prevSample
  }

  _getPrevSample (sample: Tensor, timestep: number, prevTimestep: number, modelOutput: Tensor) {
    const alphaProdT = this.alphasCumprod.data[timestep]
    const alphaProdTPrev = prevTimestep >= 0 ? this.alphasCumprod.data[prevTimestep] : this.finalAlphaCumprod

    const betaProdT = 1 - alphaProdT
    const betaProdTPrev = 1 - alphaProdTPrev
    if (this.config.prediction_type === 'v_prediction') {
      modelOutput = modelOutput.mul(Math.sqrt(alphaProdT)).add(sample.mul(Math.sqrt(betaProdT)))
    } else if (this.config.prediction_type !== 'epsilon') {
      throw new Error(`prediction_type given as ${this.config.prediction_type} must be one of 'epsilon' or 'v_prediction'`)
    }
    const sampleCoeff = Math.sqrt(alphaProdTPrev / alphaProdT)

    // corresponds to denominator of e_θ(x_t, t) in formula (9)
    const modelOutputDenomCoeff = alphaProdT * Math.sqrt(betaProdTPrev) +
      Math.sqrt(alphaProdT * betaProdT * alphaProdTPrev)

    // full formula (9)
    const prevSample = sample
      .mul(sampleCoeff)
      .sub(modelOutput.mul(alphaProdTPrev - alphaProdT).div(modelOutputDenomCoeff))

    return prevSample
  }

  // Taken from https://github.com/huggingface/diffusers/blob/d1e20be664dd8774e49d1a9d54fd71ec7cd5863c/src/diffusers/schedulers/scheduling_pndm.py#L453
  add_noise (original_samples: Tensor, noise: Tensor, timestep: number) {
    const sqrt_alpha_prod = this.alphasCumprod.data[timestep] ** 0.5
    const sqrt_one_minus_alpha_prod = (1 - this.alphasCumprod.data[timestep]) ** 0.5

    const noisy_samples = original_samples.mul(sqrt_alpha_prod).add(noise.mul(sqrt_one_minus_alpha_prod))
    return noisy_samples
  }
}

function betasForAlphaBar (numDiffusionTimesteps: number, maxBeta = 0.999) {
  function alphaBar (timeStep: number) {
    return Math.cos((timeStep + 0.008) / 1.008 * Math.PI / 2) ** 2
  }

  const betas = []
  for (let i = 0; i < numDiffusionTimesteps; i++) {
    const t1 = i / numDiffusionTimesteps
    const t2 = (i + 1) / numDiffusionTimesteps
    betas.push(Math.min(1 - alphaBar(t2) / alphaBar(t1), maxBeta))
  }
  return new Tensor(betas)
}
