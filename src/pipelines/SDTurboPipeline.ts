import { Session } from '@/backends'
import { getModelJSON } from '@/hub'
import { GetModelFileOptions } from '@/hub/common'
import { PipelineBase } from '@/pipelines/PipelineBase'
import { EulerDiscreteScheduler } from '@/schedulers/EulerDiscreteScheduler'
import { SchedulerConfig } from '@/schedulers/SchedulerBase'
import { CLIPTokenizer } from '@/tokenizers/CLIPTokenizer'
import { randomNormalTensor } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'
import { PretrainedOptions, ProgressCallback, ProgressStatus, dispatchProgress, loadModel } from './common'

export interface SDTurboInput {
  prompt: string
  seed?: string
  width?: number
  height?: number
  numInferenceSteps: number
  sdV1?: boolean
  progressCallback?: ProgressCallback
  runVaeOnEachStep?: boolean
  img2imgFlag?: boolean
  inputImage?: Float32Array
  strength?: number
}

export class SDTurboPipeline extends PipelineBase {
  declare scheduler: EulerDiscreteScheduler

  constructor (unet: Session, vaeDecoder: Session, vaeEncoder: Session, textEncoder: Session, tokenizer: CLIPTokenizer, scheduler: EulerDiscreteScheduler) {
    super()
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.vaeEncoder = vaeEncoder
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.vaeScaleFactor = 8
  }

  static createScheduler (config: SchedulerConfig) {
    return new EulerDiscreteScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
    )
  }

  static async fromPretrained (modelRepoOrPath: string, options?: PretrainedOptions) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    // order matters because WASM memory cannot be decreased. so we load the biggest one first
    const unet = await loadModel(
      modelRepoOrPath,
      'unet/model.onnx',
      opts,
    )
    const textEncoder = await loadModel(modelRepoOrPath, 'text_encoder/model.onnx', opts)
    const vaeEncoder = await loadModel(modelRepoOrPath, 'vae_encoder/model.onnx', opts)
    const vae = await loadModel(modelRepoOrPath, 'vae_decoder/model.onnx', opts)

    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = SDTurboPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new SDTurboPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
  }

  async run (input: SDTurboInput) {
    const width = input.width || 512
    const height = input.height || 512
    const batchSize = 1
    const seed = input.seed || ''
    this.scheduler.setTimesteps(input.numInferenceSteps || 1)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const promptEmbeds = await this.encodePrompt(input.prompt)

    const latentShape = [batchSize, 4, width / 8, height / 8]
    let latents = randomNormalTensor(latentShape, undefined, undefined, 'float32', seed) // Normal latents used in Text-to-Image
    const timesteps = this.scheduler.timesteps.data

    latents = latents.mul(this.scheduler.initNoiseSigma)

    let humanStep = 1
    let cachedImages: Tensor[] | null = null

    for (const step of timesteps) {
      // for some reason v1.4 takes int64 as timestep input. ideally we should get input dtype from the model
      // but currently onnxruntime-node does not give out types, only input names
      const timestep = input.sdV1
        ? new Tensor(BigInt64Array.from([BigInt(step)]))
        : new Tensor(new Float32Array([step]))
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.RunningUnet,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length,
      })
      const latentInput = this.scheduler.scaleInput(latents)

      const noise = await this.unet.run(
        { sample: latentInput, timestep, encoder_hidden_states: promptEmbeds },
      )

      const noisePred = noise.out_sample

      latents = this.scheduler.step(
        noisePred,
        step,
        latents,
      )

      if (input.runVaeOnEachStep) {
        await dispatchProgress(input.progressCallback, {
          status: ProgressStatus.RunningVae,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length,
        })
        cachedImages = await this.makeImages(latents)
      }
      humanStep++
    }

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.Done,
    })

    if (input.runVaeOnEachStep) {
      return cachedImages!
    }

    return this.makeImages(latents)
  }

  async encodeImage (inputImage: Float32Array, width: number, height: number) {
    const encoded = await this.vaeEncoder.run(
      { sample: new Tensor('float32', inputImage, [1, 3, width, height]) },
    )

    const encodedImage = encoded.latent_sample
    return encodedImage.mul(0.18215)
  }
}
