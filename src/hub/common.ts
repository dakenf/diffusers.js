import { ProgressCallback } from '@/pipelines/common'

export interface GetModelFileOptions {
  progressCallback?: ProgressCallback
  revision?: string
  returnText?: boolean
}

export interface CacheImpl {
  getModelFile (modelRepoOrPath: string, fileName: string, fatal?: boolean, options?: GetModelFileOptions): Promise<string|ArrayBuffer>
}

export function pathJoin (...parts: string[]) {
  // https://stackoverflow.com/a/55142565
  parts = parts.map((part, index) => {
    if (index) {
      part = part.replace(/^\//, '')
    }
    if (index !== parts.length - 1) {
      part = part.replace(/\/$/, '')
    }
    return part
  })
  return parts.filter(p => p !== '').join('/')
}
