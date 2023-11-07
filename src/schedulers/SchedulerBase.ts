import { linspace, range } from '@/util/Tensor'
import { betasForAlphaBar } from '@/schedulers/common'
import { Tensor } from '@xenova/transformers'

export interface SchedulerConfig {
  beta_end: number,
  beta_schedule: string,
  beta_start: number,
  clip_sample: boolean,
  num_train_timesteps: number,
  prediction_type?: 'epsilon'|'v_prediction'|'sample',
  set_alpha_to_one: boolean,
  steps_offset: number,
  trained_betas: null
}

export class SchedulerBase {
  betas: Tensor
  alphas: Tensor
  alphasCumprod!: Tensor
  finalAlphaCumprod!: number
  config: SchedulerConfig
  timesteps: Tensor
  numInferenceSteps: number = 20

  constructor (config: SchedulerConfig) {
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

    this.timesteps = range(0, config.num_train_timesteps).reverse()
    this.alphas = linspace(1, 1, config.num_train_timesteps).sub(this.betas)
    this.alphasCumprod = this.alphas.cumprod()
    this.finalAlphaCumprod = config.set_alpha_to_one ? 1.0 : this.alphasCumprod[0].data
    this.config = config
  }

  scaleModelInput (input: Tensor, timestep?: number) {
    return input
  }

  addNoise (originalSamples: Tensor, noise: Tensor, timestep: number) {
    const sqrtAlphaProd = this.alphasCumprod.data[timestep] ** 0.5
    const sqrtOneMinusAlphaProd = (1 - this.alphasCumprod.data[timestep]) ** 0.5

    return originalSamples.mul(sqrtAlphaProd).add(noise.mul(sqrtOneMinusAlphaProd))
  }
}
