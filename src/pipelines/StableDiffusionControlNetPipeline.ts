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
  controlNetImage?: Float32Array
}

export class StableDiffusionControlNetPipeline extends PipelineBase {
  declare scheduler: PNDMScheduler
  declare controlnet: Session

  constructor (unet: Session, controlnet: Session, vaeDecoder: Session, vaeEncoder: Session, textEncoder: Session, tokenizer: CLIPTokenizer, scheduler: PNDMScheduler) {
    super()
    this.unet = unet
    this.controlnet = controlnet
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
    const controlnet = await loadModel(modelRepoOrPath, 'controlnet/model.onnx', opts)
    const textEncoder = await loadModel(modelRepoOrPath, 'text_encoder/model.onnx', opts)
    const vaeEncoder = await loadModel(modelRepoOrPath, 'vae_encoder/model.onnx', opts)
    const vae = await loadModel(modelRepoOrPath, 'vae_decoder/model.onnx', opts)

    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = StableDiffusionControlNetPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new StableDiffusionControlNetPipeline(unet, controlnet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
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

    let controlnetImage = new Tensor('float32', input.controlNetImage || new Float32Array, [1, 3, 512, 512])
    controlnetImage = doClassifierFreeGuidance ? cat([controlnetImage, controlnetImage.clone()]) : controlnetImage
    const conditioningScale = new Tensor(new Float64Array([1.0]))

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

        const blocks = await this.applyControlnet(latentInput, timestep, promptEmbeds, controlnetImage, conditioningScale)

        /**
         * ControlNet blocks output shape:
         * 24549 -> [2, 320, 64, 64]
         * 24551 -> [2, 320, 32, 32]
         * 24553 -> [2, 640, 32, 32]
         * 24555 -> [2, 640, 32, 32]
         * 24557 -> [2, 640, 16, 16]
         * 24559 -> [2, 1280, 16, 16]
         * 24561 -> [2, 1280, 16, 16]
         * 24563 -> [2, 1280, 8, 8]
         * 24565 -> [2, 1280, 8, 8]
         * 24567 -> [2, 1280, 8, 8]
         * 24569 -> [2, 1280, 8, 8]
         * down_block_res_samples -> [2, 320, 64, 64]
         * mid_block_res_sample -> [2, 320, 64, 64]
         */

        /**
         * UNET down_blocks expected input shape:
         * down_block_0 -> [2, 320, 64, 64]
         * down_block_1 -> [2, 320, 64, 64]
         * down_block_2 -> [2, 320, 64, 64]
         * down_block_3 -> [2, 320, 32, 32]
         * down_block_4 -> [2, 640, 32, 32]
         * down_block_5 -> [2, 640, 32, 32]
         * down_block_6 -> [2, 640, 16, 16]
         * down_block_7 -> [2, 1280, 16, 16]
         * down_block_8 -> [2, 1280, 16, 16]
         * down_block_9 -> [2, 1280, 8, 8]
         * down_block_10 -> [2, 1280, 8, 8]
         * down_block_11 -> [2, 1280, 8, 8]
         * mid_block_additional_residual -> [2, 1280, 8, 8]
         */

        const noise = await this.unet.run(
          {
            sample: latentInput, 
            timestep, 
            encoder_hidden_states: promptEmbeds, 
            down_block_0: blocks.down_block_res_samples,
            down_block_1: blocks.mid_block_res_sample,
            down_block_2: blocks["24549"],
            down_block_3: blocks["24551"],
            down_block_4: blocks["24553"],
            down_block_5: blocks["24555"],
            down_block_6: blocks["24557"],
            down_block_7: blocks["24559"],
            down_block_8: blocks["24561"],
            down_block_9: blocks["24563"],
            down_block_10: blocks["24565"],
            down_block_11: blocks["24567"],
            mid_block_additional_residual: blocks["24569"]
          },
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

  /**
   * Applies the ControlNet model and returns the down_block_res_samples and mid_block_res_sample
   * which are used as input for the UNET model as shown in 
   * https://github.com/huggingface/diffusers/blob/29f15673ed5c14e4843d7c837890910207f72129/src/diffusers/pipelines/controlnet/pipeline_controlnet.py#L943.
   * Shapes were taken from https://docs.openvino.ai/2023.1/notebooks/235-controlnet-stable-diffusion-with-output.html#controlnet-conversion.
   * 
   * @param latentInput Latent Input that is also passed to the UNET. Expected shape: [2, 4, 64, 64]
   * @param timestep Current timestep. Expected shape: [1]
   * @param promptEmbeds Text Embeddings consisting of the prompt and negative prompt. Expected shape: [2, 77, 768]
   * @param controlnet_image Preprocessed ControlNet image. Expected shape: [2, 3, 512, 512]
   * @param conditioning_scale ControlNet Conditioning Scale. Expected shape: [1]
   */
  async applyControlnet(latentInput: Tensor, timestep: Tensor, promptEmbeds: Tensor, controlnet_image: Tensor, conditioning_scale: Tensor) {
    const blocks = await this.controlnet.run(
      {
        "sample": latentInput,
        "timestep": timestep,
        "encoder_hidden_states": promptEmbeds,
        "controlnet_cond": controlnet_image,
        "conditioning_scale": conditioning_scale
      }
    );

    return blocks;
  }

  async release () {
    await super.release()
    return this.controlnet?.release()
  }
}
