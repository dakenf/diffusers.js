import { dispatchProgress, loadModel, PretrainedOptions, ProgressCallback, ProgressStatus } from '@/pipelines/common'
import { Session } from '@/backends'
import { LCMScheduler, LCMSchedulerConfig } from '@/schedulers/LCMScheduler'
import { GetModelFileOptions } from '@/hub/common'
import { CLIPTokenizer } from '@/tokenizers/CLIPTokenizer'
import { getModelJSON } from '@/hub'
import { cat, linspace, randomNormalTensor, range } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'
import { DiffusionPipeline } from '@/pipelines/DiffusionPipeline'
import { PipelineBase } from '@/pipelines/PipelineBase'

export interface StableDiffusionInput {
  prompt: string
  negativePrompt?: string
  guidanceScale?: number
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

export class LatentConsistencyModelPipeline extends PipelineBase {
  declare public scheduler: LCMScheduler

  constructor (unet: Session, vaeDecoder: Session, vaeEncoder: Session, textEncoder: Session, tokenizer: CLIPTokenizer, scheduler: LCMScheduler) {
    super()
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.vaeEncoder = vaeEncoder
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.vaeScaleFactor = 2 ** ((this.vaeDecoder.config.block_out_channels as string[]).length - 1)
  }

  static createScheduler (config: LCMSchedulerConfig) {
    return new LCMScheduler(
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
    const scheduler = LatentConsistencyModelPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new LatentConsistencyModelPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
  }

  getWEmbedding (batchSize: number, guidanceScale: number, embeddingDim = 512) {
    let w = new Tensor('float32', new Float32Array([guidanceScale]), [1])
    w = w.mul(1000)

    const halfDim = embeddingDim / 2
    let log = Math.log(10000) / (halfDim - 1)
    let emb: Tensor = range(0, halfDim).mul(-log).exp()

    // TODO: support batch size > 1
    emb = emb.mul(w.data[0])

    return cat([emb.sin(), emb.cos()]).reshape([batchSize, embeddingDim])
  }

  async run (input: StableDiffusionInput) {
    const width = input.width || this.unet.config.sample_size as number * this.vaeScaleFactor
    const height = input.height || this.unet.config.sample_size as number * this.vaeScaleFactor
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 8.5
    const seed = input.seed || ''
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const promptEmbeds = await this.encodePrompt(input.prompt)

    let latents = this.prepareLatents(
      batchSize,
      this.unet.config.in_channels as number || 4,
      height,
      width,
      seed,
    )
    let timesteps = this.scheduler.timesteps.data

    let humanStep = 1
    let cachedImages: Tensor[] | null = null

    const wEmbedding = this.getWEmbedding(batchSize, guidanceScale, 256)
    let denoised: Tensor

    for (const step of timesteps) {
      const timestep = new Tensor(new Float32Array([step]))
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.RunningUnet,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length,
      })

      const noise = await this.unet.run(
        { sample: latents, timestep, encoder_hidden_states: promptEmbeds, timestep_cond: wEmbedding },
      );
      [latents, denoised] = this.scheduler.step(
        noise.out_sample,
        step,
        humanStep - 1,
        latents,
      )

      if (input.runVaeOnEachStep) {
        await dispatchProgress(input.progressCallback, {
          status: ProgressStatus.RunningVae,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length,
        })
        cachedImages = await this.makeImages(denoised)
      }
      humanStep++
    }

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.Done,
    })

    if (input.runVaeOnEachStep) {
      return cachedImages!
    }

    return this.makeImages(denoised)
  }
}
