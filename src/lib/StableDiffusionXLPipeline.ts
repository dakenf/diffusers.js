// @ts-ignore
import * as ORT from '@aislamov/onnxruntime-web64/webgpu';
import type { InferenceSession } from 'onnxruntime-common'
import { PNDMScheduler, SchedulerConfig } from './schedulers/PNDMScheduler'
import { CLIPTokenizer } from './tokenizers/CLIPTokenizer'
import { cat, randomNormalTensor, replaceTensors } from './Tensor'
import { Tensor } from '@xenova/transformers'
import { ProgressCallback, sessionRun } from './common'
import { getModelFile, getModelJSON } from './hub/hub'

export interface StableDiffusionXLInput {
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
  img2imgFlag: boolean
  inputImage?: Float32Array
  strength?: number
}

async function loadModel(
  modelRepoOrPath: string,
  filename: string,
  sessionOptions: ORT.InferenceSession.SessionOptions,
  opts: any
) {
  const model = await getModelFile(modelRepoOrPath, filename, true, opts)
  let weights = await getModelFile(modelRepoOrPath, filename + '_data', false, opts)

  const weightsParams = {
    externalWeights: weights,
    externalWeightsFilename: 'model.onnx_data',
  }
  const executionProviders = sessionOptions.executionProviders.map((provider) => {
    if (typeof provider === 'string') {
      return {
        name: provider,
        ...weightsParams,
      }
    }

    return {
      ...provider,
      ...weightsParams,
    }
  })

  return ORT.InferenceSession.create(
    new Uint8Array(model),
    // @ts-ignore
    { ...sessionOptions, executionProviders, }
  )
}

export class StableDiffusionXLPipeline {
  public unet: InferenceSession
  public vae: InferenceSession
  public vae_encoder: InferenceSession
  public textEncoder: InferenceSession
  public textEncoder2: InferenceSession
  public tokenizer: CLIPTokenizer
  public tokenizer2: CLIPTokenizer
  public scheduler: PNDMScheduler

  constructor(
    unet: InferenceSession,
    vae: InferenceSession,
    vae_encoder: InferenceSession,
    textEncoder: InferenceSession,
    textEncoder2: InferenceSession,
    tokenizer: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    scheduler: PNDMScheduler
  ) {
    this.unet = unet
    this.vae = vae
    this.vae_encoder = vae_encoder
    this.textEncoder = textEncoder
    this.textEncoder2 = textEncoder2
    this.tokenizer = tokenizer
    this.tokenizer2 = tokenizer2
    this.scheduler = scheduler
  }

  static createScheduler (config: SchedulerConfig) {
    return new PNDMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      }
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
      // revision: '9f697c96d42e5c09437ff14b0a2b287366ce488d',
      // local_files_only: true
    }

    const sessionOptions: ORT.InferenceSession.SessionOptions = {
      executionProviders: [executionProvider],
      executionMode: 'parallel',
      //logSeverityLevel: 0,
      //logVerbosityLevel: 0,
    }

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    const tokenizer2 = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer_2' })

    const unet = await loadModel(
      modelRepoOrPath,
      '/unet/model.onnx', {
        ...sessionOptions,
        // executionProviders: ['wasm']
      },
      opts,
    )
    const textEncoder2 = await loadModel(modelRepoOrPath, '/text_encoder_2/model.onnx', sessionOptions, opts)
    const textEncoder = await loadModel(modelRepoOrPath, '/text_encoder/model.onnx', sessionOptions, opts)
    //const vae_encoder = await InferenceSession.create(await getModelFile(modelRepoOrPath, '/vae_encoder/model.onnx', true, opts), sessionOptions)
    const vae = await loadModel(modelRepoOrPath, '/vae_decoder/model.onnx', sessionOptions, opts)

    const schedulerConfig = await getModelJSON(modelRepoOrPath, '/scheduler/scheduler_config.json', true, opts)
    const scheduler = StableDiffusionXLPipeline.createScheduler(schedulerConfig)

    progressCallback({
      step: 'Ready',
    })
    return new StableDiffusionXLPipeline(unet, vae, vae, textEncoder, textEncoder2, tokenizer, tokenizer2, scheduler)
  }

  async encodePrompt (prompt: string, tokenizer: CLIPTokenizer, textEncoder: InferenceSession) {
    const tokens = tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: true,
        max_length: tokenizer.model_max_length,
        return_tensor_dtype: 'int32'
      },
    );

    const inputIds = tokens.input_ids
    const tensor = new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length])
    // @ts-ignore
    const result = await sessionRun(textEncoder, { input_ids: tensor });
    console.log(Object.keys(result))
    const {
      last_hidden_state: lastHiddenState,
      pooler_output: poolerOutput,
      // hidden_states: hiddenStates,
      "hidden_states.11": hiddenStates,
    } = result
    return { lastHiddenState, poolerOutput, hiddenStates }
  }

  async getPromptEmbeds (prompt: string, negativePrompt: string|undefined) {
    const promptEmbeds = await this.encodePrompt(prompt, this.tokenizer, this.textEncoder)
    const negativePromptEmbeds = await this.encodePrompt(negativePrompt || '', this.tokenizer, this.textEncoder)

    const promptEmbeds2 = await this.encodePrompt(prompt, this.tokenizer2, this.textEncoder2)
    const negativePromptEmbeds2 = await this.encodePrompt(negativePrompt || '', this.tokenizer2, this.textEncoder2)

    return {
      hiddenStates: cat([
        cat([negativePromptEmbeds.hiddenStates, negativePromptEmbeds2.hiddenStates], -1),
        cat([promptEmbeds.hiddenStates, promptEmbeds2.hiddenStates], -1),
      ]),
      textEmbeds: cat([negativePromptEmbeds2.lastHiddenState, promptEmbeds2.lastHiddenState])
    }
  }

  getTimeEmbeds (width: number, height: number) {
    return new Tensor(
      'float32',
      Float32Array.from([height, width, 0, 0, height, width, height, width, 0, 0, height, width]),
      [2, 6]
    )
  }

  async run(input: StableDiffusionXLInput) {
    const width = input.width || 1024
    const height = input.height || 1024
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 5
    const seed = input.seed || ''

    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await input.progressCallback!({
      step: 'Encoding prompt...',
    })

    const promptEmbeds = await this.getPromptEmbeds(input.prompt, input.negativePrompt)

    const latentShape = [batchSize, 4, width / 8, height / 8];
    let latents = randomNormalTensor(latentShape, undefined, undefined, 'float32', seed); // Normal latents used in Text-to-Image
    let timesteps = this.scheduler.timesteps.data;

    if(input.img2imgFlag) {
      const inputImage = input.inputImage || new Float32Array()
      const strength = input.strength || 0.8

      await input.progressCallback!({
        step: 'Encoding input image...',
      })

      const image_latent = await this.encodeImage(inputImage); // Encode image to latent space

      // Taken from https://towardsdatascience.com/stable-diffusion-using-hugging-face-variations-of-stable-diffusion-56fd2ab7a265#2d1d
      const init_timestep = Math.round(input.numInferenceSteps * strength);
      const timestep = timesteps.toReversed()[init_timestep];

      latents = this.scheduler.add_noise(image_latent, latents, timestep);
      // Computing the timestep to start the diffusion loop
      const t_start = Math.max(input.numInferenceSteps - init_timestep, 0);
      timesteps = timesteps.slice(t_start);
    }

    const doClassifierFreeGuidance = guidanceScale > 1
    let humanStep = 1
    let cachedImages: Tensor[]|null = null

    const timeIds = this.getTimeEmbeds(width, height)
    const hiddenStates = promptEmbeds.hiddenStates
    const textEmbeds = promptEmbeds.textEmbeds

    for (const step of timesteps) {
      // for some reason v1.4 takes int64 as timestep input. ideally we should get input dtype from the model
      // but currently onnxruntime-node does not give out types, only input names
      const timestep = new Tensor(new Float32Array([step]))
      await input.progressCallback!({
        step: `Running unet step ${humanStep}`,
      })
      const latentInput = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents

      console.log('running', {
        sample: latentInput,
        timestep,
        encoder_hidden_states: hiddenStates,
        text_embeds: textEmbeds,
        time_ids: timeIds,
      })

      let noise = await sessionRun(
        this.unet,
        {
          sample: latentInput,
          timestep,
          encoder_hidden_states: hiddenStates,
          text_embeds: textEmbeds,
          time_ids: timeIds,
        },
      )

      console.log('noise', noise)

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
      console.log('latents', latents)

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
    }

    if (input.runVaeOnEachStep) {
      return cachedImages!
    }

    return this.makeImages(latents, batchSize, width, height)
  }

  async makeImages (latents: Tensor, batchSize: number, width: number, height: number) {
    latents = latents.mul(0.13025)

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

  // Taken from https://colab.research.google.com/github/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb#scrollTo=42d594a2-cc70-4daf-81eb-005e906118d3&line=4&uniqifier=1
  async encodeImage(input_image: Float32Array) {
    const encoded = await sessionRun(
      this.vae_encoder,
      { sample: new Tensor('float32', input_image, [1, 3, 1024, 1024]) }
    );

    const encoded_image = encoded.latent_sample;
    return encoded_image.mul(0.18215);
  }
}
