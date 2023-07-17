import { InferenceSession } from '@aislamov/onnxruntime-web64';
import { PNDMScheduler, SchedulerConfig } from './schedulers/PNDMScheduler'
// @ts-ignore
import { getModelFile, getModelJSON } from '@xenova/transformers/src/utils/hub'
import { CLIPTokenizer } from './tokenizers/CLIPTokenizer'
import { cat, randomNormalTensor, replaceTensors } from './Tensor'
import { Tensor } from '@xenova/transformers'

async function sessionRun (session: InferenceSession, inputs: Record<string, Tensor>) {
  // @ts-ignore
  const result = await session.run(inputs)
  return replaceTensors(result)
}

export interface ProgressCallbackPayload {
  images?: Tensor[]
  step: string
}

export type ProgressCallback = (cb: ProgressCallbackPayload) => Promise<void>

export interface StableDiffusionInput {
  prompt: string
  negativePrompt?: string
  guidanceScale?: number
  width?: number
  height?: number
  numInferenceSteps: number
  sdV1?: boolean
  progressCallback?: ProgressCallback
  runVaeOnEachStep?: boolean
}

export class StableDiffusionPipeline {
  public unet: InferenceSession
  public vae: InferenceSession
  public textEncoder: InferenceSession
  public tokenizer: CLIPTokenizer
  public scheduler: PNDMScheduler
  private sdVersion

  constructor(unet: InferenceSession, vae: InferenceSession, textEncoder: InferenceSession, tokenizer: CLIPTokenizer, scheduler: PNDMScheduler, sdVersion: 1|2 = 2) {
    this.unet = unet
    this.vae = vae
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.sdVersion = sdVersion
  }

  static async createScheduler (config: SchedulerConfig) {
    return new PNDMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
      config.num_train_timesteps,
      config.beta_start,
      config.beta_end,
      config.beta_schedule,
    )
  }

  async sleep (ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  static async fromPretrained(executionProvider: 'wasm'|'webgpu'|'cpu'|'cuda'|'directml' = 'cpu', modelRepoOrPath: string, progressCallback: ProgressCallback) {
    const hubProgressCallback = (data: { status: string, file: string, progress: number }) => {
      if (data.status === 'progress') {
        progressCallback({
          step: `Downloading ${data.file}, ${Math.round(data.progress)}%`
        })
      } else {
        progressCallback({
          step: `Downloaded ${data.file}`
        })
      }
    }

    const opts = {
      progress_callback: hubProgressCallback,
    }
    const sessionOption: InferenceSession.SessionOptions = { executionProviders: [executionProvider] }
    const unet = await InferenceSession.create(await getModelFile(modelRepoOrPath, '/unet/model.onnx', true, opts), { executionProviders: ['wasm'] })
    const textEncoder = await InferenceSession.create(await getModelFile(modelRepoOrPath, '/text_encoder/model.onnx', true, opts), sessionOption)
    const vae = await InferenceSession.create(await getModelFile(modelRepoOrPath, '/vae_decoder/model.onnx', true, opts), sessionOption)

    const schedulerConfig = await getModelJSON(modelRepoOrPath, '/scheduler/scheduler_config.json', true, opts)
    const scheduler = await StableDiffusionPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, opts)
    progressCallback({
      step: 'Ready',
    })
    return new StableDiffusionPipeline(unet, vae, textEncoder, tokenizer, scheduler, 2)
  }

  async encodePrompt (prompt: string): Promise<Tensor> {
    const tokens = this.tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: true,
        max_length: this.tokenizer.model_max_length,
        return_tensor_dtype: 'int32'
      },
    );

    const inputIds = tokens.input_ids
    // @ts-ignore
    const encoded = await sessionRun(this.textEncoder, { input_ids: new Tensor(Int32Array.from(inputIds.flat()), [1, inputIds.length]) });
    return encoded.last_hidden_state
  }

  async getPromptEmbeds (prompt: string, negativePrompt: string|undefined) {
    const promptEmbeds = await this.encodePrompt(prompt)
    const negativePromptEmbeds = await this.encodePrompt(negativePrompt || '')

    return cat([negativePromptEmbeds, promptEmbeds])
  }

  async run(input: StableDiffusionInput) {
    const width = input.width || 512
    const height = input.height || 512
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 7.5
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)
    await input.progressCallback!({
      step: 'Encoding prompt...',
    })

    const promptEmbeds = await this.getPromptEmbeds(input.prompt, input.negativePrompt)

    const latentShape = [batchSize, 4, width / 8, height / 8]
    let latents = randomNormalTensor(latentShape, undefined, undefined, 'float32')

    const doClassifierFreeGuidance = guidanceScale > 1
    let humanStep = 1
    let cachedImages: Tensor[]|null = null
    for (const step of this.scheduler.timesteps.data) {
      // for some reason v1.4 takes int64 as timestep input. ideally we should get input dtype from the model
      // but currently onnxruntime-node does not give out types, only input names
      const timestep = input.sdV1
        ? new Tensor(BigInt64Array.from([BigInt(step)]))
        : new Tensor(new Float32Array([step]))
      await input.progressCallback!({
        step: `Running unet step ${humanStep}`,
      })
      // sleep to update UI
      await this.sleep(100)
      const latentInput = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents

      let noise = await sessionRun(
        this.unet,
        { sample: await latentInput, timestep, encoder_hidden_states: promptEmbeds },
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

      if (input.progressCallback) {
        if (input.runVaeOnEachStep) {
          await input.progressCallback({
            step: `Running vae...`,
          })
          cachedImages = await this.makeImages(latents, batchSize, width, height)
          await input.progressCallback({
            step: 'Finished step ' + humanStep,
            images: cachedImages,
          })
        } else {
          await input.progressCallback({
            step: 'Finished step ' + humanStep,
          })
        }
      }
      humanStep++
      // sleep to update UI
      await this.sleep(500)
    }

    if (input.runVaeOnEachStep) {
      return cachedImages!
    }

    return this.makeImages(latents, batchSize, width, height)
  }

  async makeImages (latents: Tensor, batchSize: number, width: number, height: number) {
    latents = latents.mul(1 / 0.18215)

    const decoded = await sessionRun(
      this.vae,
      { latent_sample: latents }
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
}
