import * as tf from '@tensorflow/tfjs'
import { InferenceSession, Tensor } from 'onnxruntime-web';
import Tokenizer from './tokenizer'
import { PNDMScheduler } from './schedulers/PNDMScheduler'

function extendArray(arr: number[], length: number) {
  return arr.concat(Array(length - arr.length).fill(0));
}

function extendBigArray(arr: BigInt[], length: number) {
  return arr.concat(Array(length - arr.length).fill(0));
}

interface SchedulerConfig {
  "beta_end": number,
  "beta_schedule": string,
  "beta_start": number,
  "clip_sample": boolean,
  "num_train_timesteps": number,
  prediction_type?: "epsilon"|"v-preditcion",
  "set_alpha_to_one": boolean,
  "skip_prk_steps": boolean,
  "steps_offset": number,
  "trained_betas": null
}

export interface ProgressCallback {
  images: tf.Tensor2D[]
  step: string
}

export class StableDiffusionPipeline {
  public unet: InferenceSession
  public vae: InferenceSession
  public textEncoder?: InferenceSession
  public tokenizer: Tokenizer
  public scheduler: PNDMScheduler
  private sdVersion
  private static sessionOptions: InferenceSession.SessionOptions
  private static searchPath: string

  constructor(unet: InferenceSession, vae: InferenceSession, textEncoder: InferenceSession|undefined, tokenizer: Tokenizer, scheduler: PNDMScheduler, sdVersion: 1|2 = 2) {
    this.unet = unet
    this.vae = vae
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.sdVersion = sdVersion
  }

  static async createScheduler (config: SchedulerConfig) {
    const scheduler = new PNDMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
      config.num_train_timesteps,
      config.beta_start,
      config.beta_end,
      config.beta_schedule,
    )
    await scheduler.setAlphasCumprod()

    return scheduler
  }

  static async fromPretrained(executionProvider: 'cpu'|'cuda'|'directml'|'wasm'|'webnn'|'webgpu'|'webgl' = 'cpu', modelRepoOrPath: string, revision?: string, sdVersion: 1|2 = 2) {
    let searchPath = modelRepoOrPath
    // let's check in the cache if path does not exist
    // if (!await fileExists(`${searchPath}/text_encoder/model.onnx`) && searchPath[0] !== '.' && searchPath[0] !== '/') {
    //   searchPath = MODEL_CACHE_DIR + modelRepoOrPath
    //   if (!await fileExists(`${searchPath}/text_encoder/model.onnx`)) {
    //     console.log(`Model not found in cache dir ${searchPath}, downloading from hub...`)
    //     await StableDiffusionPipeline.downloadFromHub(searchPath, modelRepoOrPath, revision)
    //   }
    // }
    //
    // if (!await fileExists(`${searchPath}/text_encoder/model.onnx`)) {
    //   throw new Error("Could not find model files. Maybe you are not using onnx version")
    // }
    this.searchPath = searchPath;

    const sessionOption: InferenceSession.SessionOptions = {
      executionProviders: [executionProvider],
      executionMode: 'sequential',
      // enableCpuMemArena: true,
      // logSeverityLevel: 0,
      // logVerbosityLevel: 0,
      // extra: {
      //   session: {
      //     use_ort_model_bytes_directly: "0",
      //     use_ort_model_bytes_for_initializers: "0",
      //   }
      // }
    }
    this.sessionOptions = sessionOption

    const unet = await InferenceSession.create(`${searchPath}/unet/model.onnx`, sessionOption)
    const textEncoder = await InferenceSession.create(`${StableDiffusionPipeline.searchPath}/text_encoder/model.onnx`, StableDiffusionPipeline.sessionOptions)
    const vae = await InferenceSession.create(`${searchPath}/vae_decoder/model.onnx`, sessionOption)

    const schedulerConfig = await (await fetch(`${searchPath}/scheduler/scheduler_config.json`)).json()
    const scheduler = await StableDiffusionPipeline.createScheduler(schedulerConfig)

    const merges = await (await fetch(`${searchPath}/tokenizer/merges.txt`)).text()
    const tokenizerConfig = await (await fetch(`${searchPath}/tokenizer/tokenizer_config.json`)).json()
    const vocab = await (await fetch(`${searchPath}/tokenizer/vocab.json`)).json()
    return new StableDiffusionPipeline(unet, vae, textEncoder, new Tokenizer(merges, tokenizerConfig, vocab.toString()), scheduler, sdVersion)
  }

  async encodePrompt (prompt: string): Promise<Tensor> {
    const tokens = this.tokenizer.encode(prompt)
    const tensorTokens = new Tensor('int32', Int32Array.from(extendArray([49406, ...tokens.slice(0, this.tokenizer.tokenMaxLen - 2), 49407], 77)), [1, 77])
    const encoded = await this.textEncoder?.run({ input_ids: tensorTokens })
    return encoded?.last_hidden_state as Tensor
  }

  async getPromptEmbeds (prompt: string, negativePrompt: string|undefined) {
    if (!this.textEncoder) {
      this.textEncoder = await InferenceSession.create(`${StableDiffusionPipeline.searchPath}/text_encoder/model.onnx`, StableDiffusionPipeline.sessionOptions)
    }
    const promptEmbeds = await this.encodePrompt(prompt)
    const negativePromptEmbeds = await this.encodePrompt(negativePrompt || '')

    const newShape = [...promptEmbeds.dims]
    newShape[0] = 2
    return new Tensor('float32', [...negativePromptEmbeds.data as unknown as number[], ...promptEmbeds.data as unknown as number[]], newShape)
  }

  async makeImages (latents: tf.Tensor, batchSize: number, width: number, height: number) {
    latents = latents.clone().mul(tf.tensor(1).div(0.18215))
    const decoded = await this.vae.run({ latent_sample: new Tensor('float32', await latents.data(), [1, 4, width / 8, height / 8]) })
    const decodedTensor = tf.tensor(decoded.sample.data as Float32Array, decoded.sample.dims as number[], decoded.sample.type as 'float32')
    return decodedTensor
      .div(2)
      .add(0.5)
      .mul(255).round().clipByValue(0, 255).cast('int32')
      .transpose([0, 2, 3, 1])
      .split(batchSize, 0)
  }

  async run (prompt: string, negativePrompt: string|undefined, batchSize: number, guidanceScale: number, numInferenceSteps: number, runVaeOnEachStep = false, callback: (i: ProgressCallback) => Promise<void>) {
    const width = 512
    const height = 512
    if (batchSize != 1) {
      throw new Error('Currently only batch size of 1 is supported.')
    }

    this.scheduler.setTimesteps(numInferenceSteps)

    let promptEmbeds
    const saved = localStorage.getItem('prompt1');
    if (saved) {
      promptEmbeds = JSON.parse(saved)
      const array = Float32Array.from(Object.values(promptEmbeds.data))
      promptEmbeds = new Tensor('float32', array, promptEmbeds.dims)
    } else {
      promptEmbeds = await this.getPromptEmbeds(prompt, negativePrompt)
      localStorage.setItem('prompt', JSON.stringify(promptEmbeds))
    }

    // @ts-ignore
    // await this.textEncoder.release()

    const latentShape = [batchSize, 4, width / 8, height / 8]
    let latents = tf.randomNormal(latentShape, undefined, undefined, 'float32')

    const doClassifierFreeGuidance = guidanceScale > 1
    for (const step of (await this.scheduler.timesteps.data() as unknown as number[])) {
      console.log('step', step)
      // for some reason v1.4 takes int64 as timestep input. ideally we should get input dtype from the model
      const timestep = this.sdVersion == 2
        ? new Tensor('float32', [step])
        : new Tensor(BigInt64Array.from([BigInt(step)]), [1])

      const latentInputTf = doClassifierFreeGuidance ? latents.concat(latents.clone()) : latents
      const latentInput = new Tensor(await latentInputTf.data(), latentInputTf.shape)

      let noise = await this.unet.run(
        { sample: latentInput, timestep, encoder_hidden_states: promptEmbeds },
      )
      let noisePred = Object.values(noise)[0].data as Float32Array

      let noisePredTf
      if (doClassifierFreeGuidance) {
        const len = Object.values(noise)[0].data.length / 2
        const [noisePredUncond, noisePredText] = [
          tf.tensor(noisePred.slice(0, len), latentShape, 'float32'),
          tf.tensor(noisePred.slice(len, len * 2), latentShape, 'float32'),
        ]
        noisePredTf = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(guidanceScale))
      } else {
        noisePredTf = tf.tensor(noisePred, latentShape, 'float32')
      }

      const schedulerOutput = this.scheduler.step(
        noisePredTf,
        step,
        latents,
      )
      latents = schedulerOutput
      if (callback) {
        if (runVaeOnEachStep) {
          const images = await this.makeImages(latents, batchSize, width, height)
          await callback({
            step: 'Finished step',
            images: images.map(t => t.squeeze([0])),
          })
        }
      }
    }

    return this.makeImages(latents, batchSize, width, height)
  }
}
