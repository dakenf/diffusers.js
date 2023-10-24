import { PNDMScheduler, SchedulerConfig } from '@/schedulers/PNDMScheduler'
import { CLIPTokenizer } from '../tokenizers/CLIPTokenizer'
import { cat, randomNormalTensor } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'
import { dispatchProgress, PretrainedOptions, ProgressCallback, ProgressStatus } from './common'
import { getModelFile, getModelJSON } from '@/hub'
import { Session } from '@/backends'
import { GetModelFileOptions } from '@/hub/common'

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

export class StableDiffusionPipeline {
  public unet: Session
  public vaeDecoder: Session
  public vaeEncoder: Session
  public textEncoder: Session
  public tokenizer: CLIPTokenizer
  public scheduler: PNDMScheduler
  private sdVersion

  constructor (unet: Session, vaeDecoder: Session, vaeEncoder: Session, textEncoder: Session, tokenizer: CLIPTokenizer, scheduler: PNDMScheduler, sdVersion: 1 | 2 = 2) {
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.vaeEncoder = vaeEncoder
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.sdVersion = sdVersion
  }

  static createScheduler (config: SchedulerConfig) {
    return new PNDMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
    )
  }

  async sleep (ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  static async fromPretrained (modelRepoOrPath: string, options?: PretrainedOptions) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    // order matters because WASM memory cannot be decreased. so we load the biggset one first
    const unet = await Session.create(await getModelFile(modelRepoOrPath, 'unet/model.onnx', true, opts))
    const textEncoder = await Session.create(await getModelFile(modelRepoOrPath, 'text_encoder/model.onnx', true, opts))
    const vaeEncoder = await Session.create(await getModelFile(modelRepoOrPath, 'vae_encoder/model.onnx', true, opts))
    const vae = await Session.create(await getModelFile(modelRepoOrPath, 'vae_decoder/model.onnx', true, opts))

    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = StableDiffusionPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new StableDiffusionPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler, 2)
  }

  async encodePrompt (prompt: string): Promise<Tensor> {
    const tokens = this.tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: true,
        max_length: this.tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )

    const inputIds = tokens.input_ids
    // @ts-ignore
    const encoded = await this.textEncoder.run({ input_ids: new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length]) })
    return encoded.last_hidden_state
  }

  async getPromptEmbeds (prompt: string, negativePrompt: string | undefined) {
    const promptEmbeds = await this.encodePrompt(prompt)
    const negativePromptEmbeds = await this.encodePrompt(negativePrompt || '')

    return cat([negativePromptEmbeds, promptEmbeds])
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

      latents = this.scheduler.add_noise(imageLatent, latents, timestep)
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

  async makeImages (latents: Tensor) {
    latents = latents.mul(1 / 0.18215)

    const decoded = await this.vaeDecoder.run(
      { latent_sample: latents },
    )

    const images = decoded.sample
      .div(2)
      .add(0.5)
    // .mul(255)
    // .round()
    // .clipByValue(0, 255)
    // .transpose(0, 2, 3, 1)
    return [images]
  }

  async encodeImage (inputImage: Float32Array, width: number, height: number) {
    const encoded = await this.vaeEncoder.run(
      { sample: new Tensor('float32', inputImage, [1, 3, width, height]) },
    )

    const encodedImage = encoded.latent_sample
    return encodedImage.mul(0.18215)
  }
}
