import { InferenceSession } from 'onnxruntime-common'
import { Tensor } from '@xenova/transformers'
import { replaceTensors } from '@/util/Tensor'

export async function sessionRun (session: InferenceSession, inputs: Record<string, Tensor>) {
  // @ts-ignore
  const result = await session.run(inputs)
  return replaceTensors(result)
}

export interface PretrainedOptions {
  revision?: string
  progressCallback?: ProgressCallback
}

export const enum ProgressStatus {
  Downloading = 'Downloading',
  Ready = 'Ready',
  Error = 'Error',
  EncodingImg2Img = 'EncodingImg2Img',
  EncodingPrompt = 'EncodingPrompt',
  RunningUnet = 'RunningUnet',
  RunningVae = 'RunningVae',
  Done = 'Done',
}

interface ProgressDownloadStatus {
  file: string
  size: number
  downloaded: number
}

export interface ProgressCallbackPayload {
  status: ProgressStatus
  downloadStatus?: ProgressDownloadStatus
  images?: Tensor[]
  statusText?: string
  unetTotalSteps?: number
  unetTimestep?: number
}

export type ProgressCallback = (cb: ProgressCallbackPayload) => Promise<void>

function setStatusText (payload: ProgressCallbackPayload) {
  switch (payload.status) {
    case ProgressStatus.Downloading:
      return `Downloading ${payload.downloadStatus!.file} (${Math.round(payload.downloadStatus!.downloaded / payload.downloadStatus!.size * 100)}%)`
    case ProgressStatus.EncodingImg2Img:
      return `Encoding input image`
    case ProgressStatus.EncodingPrompt:
      return `Encoding prompt`
    case ProgressStatus.RunningUnet:
      return `Running UNet (${payload.unetTimestep!}/${payload.unetTotalSteps!})`
    case ProgressStatus.RunningVae:
      return `Running VAE`
    case ProgressStatus.Done:
      return `Done`
    case ProgressStatus.Ready:
      return `Ready`
  }

  return ''
}

export function dispatchProgress (cb: ProgressCallback, payload: ProgressCallbackPayload) {
  if (!payload.statusText) {
    payload.statusText = setStatusText(payload)
  }
  if (cb) {
    return cb(payload)
  }
}
