import * as tf from '@tensorflow/tfjs'


export class PNDMScheduler {
  betas: tf.Tensor1D;
  alphas: tf.Tensor1D;
  alphasCumprod!: Float32Array;
  finalAlphaCumprod!: number;
  initNoiseSigma: number;
  pndmOrder: number;
  curModelOutput: number;
  counter: number;
  curSample: tf.Tensor|null;
  ets: tf.Tensor[];
  numInferenceSteps: number = 20;
  timesteps: tf.Tensor;
  prkTimesteps: tf.Tensor;
  plmsTimesteps: tf.Tensor;
  config: any; // Define your config type

  constructor(
    config: any,
    numTrainTimesteps: number = 1000,
    betaStart: number = 0.00085,
    betaEnd: number = 0.012,
    betaSchedule: string = "scaled_linear",
    trainedBetas: tf.Tensor1D | null = null,
    skipPrkSteps: boolean = false,
    setAlphaToOne: boolean = false,
    predictionType: string = "epsilon",
    stepsOffset: number = 0
  ) {
    this.config = config;

    if (trainedBetas !== null) {
      // this.betas = nj.array(trainedBetas);
      this.betas = tf.linspace(betaStart, betaEnd, numTrainTimesteps)
    } else if (betaSchedule === "linear") {
      this.betas = tf.linspace(betaStart, betaEnd, numTrainTimesteps)
    } else if (betaSchedule === "scaled_linear") {
      this.betas = tf.linspace(betaStart ** 0.5, betaEnd ** 0.5, numTrainTimesteps).pow(2);
    } else if (betaSchedule === "squaredcos_cap_v2") {
      // this.betas = betasForAlphaBar(numTrainTimesteps);
      this.betas = tf.linspace(betaStart, betaEnd, numTrainTimesteps)
    } else {
      throw new Error(`${betaSchedule} does is not implemented for ${this.constructor}`);
    }

    this.alphas = tf.sub(1.0, this.betas)

    this.initNoiseSigma = 1.0;
    this.pndmOrder = 4;

    // running values
    this.curModelOutput = 0;
    this.counter = 0;
    this.curSample = null;
    this.ets = []

    // setable values
    this.timesteps = tf.tensor(Array.from({ length: numTrainTimesteps }, (_, i) => i).reverse())
    this.prkTimesteps = tf.tensor([])
    this.plmsTimesteps = tf.tensor([])
  }

  async setAlphasCumprod () {
    this.alphasCumprod = await tf.cumprod(this.alphas).data() as Float32Array
    const setAlphaToOne = false
    this.finalAlphaCumprod = setAlphaToOne ? 1.0 : this.alphasCumprod[0]
  }

  setTimesteps (numInferenceSteps: number) {
    this.numInferenceSteps = numInferenceSteps;
    const stepRatio = ~~(this.config.num_train_timesteps / this.numInferenceSteps)
    this.timesteps = tf.range(0, numInferenceSteps).mul(stepRatio).round();
    this.timesteps = this.timesteps.add(this.config.steps_offset);

    if (this.config.skip_prk_steps) {
      this.prkTimesteps = tf.tensor([]);
      const size = this.timesteps.size;
      this.plmsTimesteps = tf.concat([
        this.timesteps.slice(0, size-1),
        this.timesteps.slice(size-2, 1),
        this.timesteps.slice(size-1, 1)
      ]).reverse().clone();
    } else {
      const prkTimesteps = this.timesteps.slice(-this.pndmOrder)
        .tile([2])
        .add(
          tf.tensor([0, this.config.num_train_timesteps / numInferenceSteps / 2]).tile([this.pndmOrder])
        );
      this.prkTimesteps = prkTimesteps.slice(0, -1).tile([2]).slice(1, -1).reverse().clone();
      this.plmsTimesteps = this.timesteps.slice(0, -3).reverse().clone();
    }

    const timesteps = tf.concat([this.prkTimesteps, this.plmsTimesteps]).asType('int32');
    this.timesteps = timesteps;
    this.ets = []
    this.counter = 0;
    this.curModelOutput = 0;
  }

  step(
    modelOutput: tf.Tensor,
    timestep: number,
    sample: tf.Tensor,
    returnDict: boolean = true
  ): tf.Tensor {
    if (this.counter < this.prkTimesteps.shape[0] && !this.config.skip_prk_steps) {
      return this.stepPrk(modelOutput, timestep, sample, returnDict);
    } else {
      return this.stepPlms(modelOutput, timestep, sample, returnDict);
    }
  }

  stepPrk(
    modelOutput: tf.Tensor,
    timestep: number,
    sample: tf.Tensor,
    returnDict: boolean = true
  ): tf.Tensor {
    throw new Error("Not implemented")
    // if (this.numInferenceSteps === null) {
    //   throw new Error(
    //     "Number of inference steps is 'null', you need to run 'setTimesteps' after creating the scheduler"
    //   );
    // }

    // const diffToPrev = this.counter % 2 === 0 ? this.config.num_train_timesteps / this.numInferenceSteps / 2 : 0;
    // const prevTimestep = timestep - diffToPrev;
    // timestep = this.prkTimesteps.get(this.counter / 4 * 4);
    //
    // if (this.counter % 4 === 0) {
    //   this.curModelOutput = this.curModelOutput.add(modelOutput.mul(1 / 6));
    //   this.ets.push(modelOutput);
    //   this.curSample = sample;
    // } else if ((this.counter - 1) % 4 === 0) {
    //   this.curModelOutput = this.curModelOutput.add(modelOutput.mul(1 / 3));
    // } else if ((this.counter -2 ) % 4 === 0) {
    //   this.curModelOutput = this.curModelOutput.add(modelOutput.mul(1 / 3));
    // } else if ((this.counter - 3) % 4 === 0) {
    //   modelOutput = this.curModelOutput.add(modelOutput.mul(1 / 6));
    //   this.curModelOutput = 0;
    // }
    // const curSample = this.curSample !== null ? this.curSample : sample;
    // const prevSample = this._getPrevSample(curSample, timestep, prevTimestep, modelOutput);
    // this.counter += 1;

    return modelOutput;
  }

  stepPlms(
    modelOutput: tf.Tensor,
    timestep: number,
    sample: tf.Tensor,
    returnDict: boolean = true
  ): tf.Tensor {
    let prevTimestep = timestep - ~~(this.config.num_train_timesteps / this.numInferenceSteps)

    if (this.counter !== 1) {
      this.ets = this.ets.slice(-3);
      this.ets.push(modelOutput);
    } else {
      prevTimestep = timestep;
      timestep = timestep + ~~(this.config.num_train_timesteps / this.numInferenceSteps)
    }

    if (this.ets.length === 1 && this.counter === 0) {
      modelOutput = modelOutput;
      this.curSample = sample;
    } else if (this.ets.length === 1 && this.counter === 1) {
      modelOutput = modelOutput.add(this.ets[this.ets.length - 1]).div(2);
      sample = this.curSample!;
      this.curSample = null;
    } else if (this.ets.length === 2) {
      modelOutput = this.ets[this.ets.length - 1].mul(3)
        .sub(this.ets[this.ets.length - 2])
        .div(2)
    } else if (this.ets.length === 3) {
      modelOutput =
        this.ets[this.ets.length - 1].mul(23)
        .sub(
          this.ets[this.ets.length - 2].mul(16)
        )
        .add(
          this.ets[this.ets.length - 3].mul(5)
        )
        .div(12)
    } else {
      modelOutput =
        this.ets[this.ets.length - 1].mul(55)
        .sub(
          this.ets[this.ets.length - 2].mul(59)
        )
        .add(
          this.ets[this.ets.length - 3].mul(37)
        )
        .sub(
          this.ets[this.ets.length - 4].mul(9)
        ).mul(1 / 24);
    }
    // modelOutput.data().then((data) => { console.log('modelOutput', modelOutput.shape, data) })
    const prevSample = this._getPrevSample(sample, timestep, prevTimestep, modelOutput);
    this.counter += 1;

    return prevSample
  }

  _getPrevSample(sample: tf.Tensor, timestep: number, prevTimestep: number, modelOutput: tf.Tensor): tf.Tensor {
    const alphaProdT = this.alphasCumprod[timestep]
    const alphaProdTPrev = prevTimestep >= 0 ? this.alphasCumprod[prevTimestep] : this.finalAlphaCumprod;

    const betaProdT = 1 - alphaProdT;
    const betaProdTPrev = 1 - alphaProdTPrev;
    if (this.config.prediction_type === 'v_prediction') {
      modelOutput = modelOutput.mul(Math.sqrt(alphaProdT)).add(sample.mul(Math.sqrt(betaProdT)));
    } else if (this.config.prediction_type !== 'epsilon') {
      throw new Error(`prediction_type given as ${this.config.prediction_type} must be one of 'epsilon' or 'v_prediction'`);
    }
    const sampleCoeff = Math.sqrt(alphaProdTPrev / alphaProdT)

    // corresponds to denominator of e_θ(x_t, t) in formula (9)
    const modelOutputDenomCoeff = alphaProdT * Math.sqrt(betaProdTPrev)
      + Math.sqrt(alphaProdT * betaProdT * alphaProdTPrev)

    // full formula (9)
    const prevSample = sample
      .mul(sampleCoeff)
      .sub(modelOutput.mul(alphaProdTPrev - alphaProdT).div(modelOutputDenomCoeff));

    return prevSample;
  }

}

// function betasForAlphaBar(numDiffusionTimesteps: number, maxBeta = 0.999): nj.NdArray {
//   function alphaBar(timeStep: number): number {
//     return Math.cos((timeStep + 0.008) / 1.008 * Math.PI / 2) ** 2;
//   }
//
//   const betas = [];
//   for (let i = 0; i < numDiffusionTimesteps; i++) {
//     const t1 = i / numDiffusionTimesteps;
//     const t2 = (i + 1) / numDiffusionTimesteps;
//     betas.push(Math.min(1 - alphaBar(t2) / alphaBar(t1), maxBeta));
//   }
//   return nj.array(betas);
// }
