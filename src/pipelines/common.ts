import { InferenceSession } from 'onnxruntime-common'
import { Tensor } from '@xenova/transformers'
import { replaceTensors } from '../lib/Tensor'

export async function sessionRun (session: InferenceSession, inputs: Record<string, Tensor>) {
  // @ts-ignore
  const result = await session.run(inputs)
  return replaceTensors(result)
}

export interface ProgressCallbackPayload {
  images?: Tensor[]
  step: string
  unetTimestep?: number
}

export type ProgressCallback = (cb: ProgressCallbackPayload) => Promise<void>
