import { SchedulerBase, SchedulerConfig } from '@/schedulers/SchedulerBase'
import { cat, interp, linspace, randomNormalTensor, range } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'

/**
 * Euler discrete scheduler
 */
export class EulerDiscreteScheduler extends SchedulerBase {
  sigmas: Tensor
  stepIndex: number = 0

  constructor (
    config: SchedulerConfig,
  ) {
    super(config)
    this.betas = linspace(
      config.beta_start ** 0.5,
      config.beta_end ** 0.5,
      config.num_train_timesteps,
    ).pow(2)

    this.alphas = linspace(1, 1, config.num_train_timesteps).sub(this.betas)
    this.alphasCumprod = this.alphas.cumprod()

    this.sigmas = linspace(1, 1, config.num_train_timesteps)
      .sub(this.alphasCumprod)
      .div(this.alphasCumprod)
      .sqrt()
    this.timesteps = linspace(
      0,
      config.num_train_timesteps - 1,
      config.num_train_timesteps,
    ).reverse()

    this.sigmas = cat([
      this.sigmas.reverse(),
      new Tensor(this.sigmas.type, [0], [1]),
    ])

    this.config = config
  }

  setTimesteps (numInferenceSteps: number) {
    this.numInferenceSteps = numInferenceSteps

    const stepRatio = ~~(
      this.config.num_train_timesteps / this.numInferenceSteps
    )
    this.timesteps = range(1, numInferenceSteps + 1)
      .reverse()
      .mul(stepRatio)
      .round()
    this.timesteps = this.timesteps.sub(1)

    this.sigmas = linspace(1, 1, this.config.num_train_timesteps)
      .sub(this.alphasCumprod)
      .div(this.alphasCumprod)
      .sqrt()
    this.sigmas = interp(
      this.timesteps,
      range(0, this.sigmas.data.length),
      this.sigmas,
    )

    this.sigmas = cat([this.sigmas, new Tensor(this.sigmas.type, [0], [1])])

    this.stepIndex = 0
  }

  scaleInput (input: Tensor) {
    const sigma = this.sigmas.data[this.stepIndex]
    const scale = (sigma ** 2 + 1) ** 0.5
    return input.div(scale)
  }

  get initNoiseSigma () {
    return Math.max(...this.sigmas.data)
  }

  step (
    modelOutput: Tensor,
    timestep: number,
    sample: Tensor,
    s_churn: number = 0.0,
    s_tmin: number = 0.0,
    s_tmax: number = Infinity,
    s_noise: number = 1.0,
  ) {
    if (this.numInferenceSteps === null) {
      throw new Error(
        "Number of inference steps is 'null', you need to run 'setTimesteps' after creating the scheduler",
      )
    }

    const sigma = this.sigmas.data[this.stepIndex]

    // Get gama with the equivalent of this python code
    let gamma = 0.0
    if (s_tmin <= sigma && sigma <= s_tmax) {
      gamma = Math.min(
        s_churn / (this.sigmas.data.length - 1),
        Math.sqrt(2) - 1,
      )
    }

    const noise = randomNormalTensor(modelOutput.dims)

    const eps = noise.mul(s_noise)
    const sigma_hat = sigma * (gamma + 1)

    if (gamma > 0) {
      sample = sample.add(eps.mul(sigma_hat ** 2 - sigma ** 2).sqrt())
    }

    // # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    // # config.prediction_type == "epsilon":
    const denoised = sample.sub(modelOutput.mul(sigma_hat))

    // 2. Convert to an ODE derivative
    const derivative = sample.sub(denoised).div(sigma_hat)

    const dt = this.sigmas.data[this.stepIndex + 1] - sigma_hat

    const prevSample = sample.add(derivative.mul(dt))

    this.stepIndex++

    return prevSample
  }
}
