import { PNDMScheduler, PNDMSchedulerConfig } from '@/schedulers/PNDMScheduler'
import { CLIPTokenizer } from '@/tokenizers/CLIPTokenizer'
import { cat, randomNormalTensor } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'
import { dispatchProgress, loadModel, PretrainedOptions, ProgressCallback, ProgressStatus } from './common'
import { getModelJSON } from '@/hub'
import { Session } from '@/backends'
import { GetModelFileOptions } from '@/hub/common'
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

export class StableDiffusionPipeline extends PipelineBase {
  declare scheduler: PNDMScheduler

  constructor (unet: Session, vaeDecoder: Session, vaeEncoder: Session, textEncoder: Session, tokenizer: CLIPTokenizer, scheduler: PNDMScheduler) {
    super()
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.vaeEncoder = vaeEncoder
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.vaeScaleFactor = 8
  }

  static createScheduler (config: PNDMSchedulerConfig) {
    return new PNDMScheduler(
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
    const scheduler = StableDiffusionPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new StableDiffusionPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
  }

  async run (input: StableDiffusionInput) {
    const width = input.width || 512
    const height = input.height || 512
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 7.5
    const seed = input.seed || ''
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const promptEmbeds = await this.getPromptEmbeds(input.prompt, input.negativePrompt)

    const latentShape = [batchSize, 4, width / 8, height / 8]
    let latents = randomNormalTensor(latentShape, undefined, undefined, 'float32', seed) // Normal latents used in Text-to-Image
    let timesteps = this.scheduler.timesteps.data

    if (input.img2imgFlag) {
      const inputImage = input.inputImage || new Float32Array()
      const strength = input.strength || 0.8

      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.EncodingImg2Img,
      })

      const imageLatent = await this.encodeImage(inputImage, input.width, input.height) // Encode image to latent space

      // Taken from https://towardsdatascience.com/stable-diffusion-using-hugging-face-variations-of-stable-diffusion-56fd2ab7a265#2d1d
      const initTimestep = Math.round(input.numInferenceSteps * strength)
      const timestep = timesteps.toReversed()[initTimestep]

      latents = this.scheduler.addNoise(imageLatent, latents, timestep)
      // Computing the timestep to start the diffusion loop
      const tStart = Math.max(input.numInferenceSteps - initTimestep, 0)
      timesteps = timesteps.slice(tStart)
    }

    const doClassifierFreeGuidance = guidanceScale > 1
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
      const latentInput = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents

      const noise = await this.unet.run(
        { sample: latentInput, timestep, encoder_hidden_states: promptEmbeds },
      )

      let noisePred = noise.out_sample
      if (doClassifierFreeGuidance) {
        const [noisePredUncond, noisePredText] = [
          noisePred.slice([0, 1]),
          noisePred.slice([1, 2]),
        ]
        noisePred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(guidanceScale))
      }

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
