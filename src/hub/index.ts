import { CacheImpl, GetModelFileOptions } from '@/hub/common'

let cacheImpl: CacheImpl = null

export function setCacheImpl (impl: CacheImpl) {
  cacheImpl = impl
}

export async function getModelFile (modelRepoOrPath: string, fileName: string, fatal = true, options: GetModelFileOptions = {}) {
  return cacheImpl.getModelFile(modelRepoOrPath, fileName, fatal, options)
}

export function getModelTextFile (modelPath: string, fileName: string, fatal: boolean, options: GetModelFileOptions) {
  return getModelFile(modelPath, fileName, fatal, { ...options, returnText: true }) as Promise<string>
}

export async function getModelJSON (modelPath: string, fileName: string, fatal = true, options: GetModelFileOptions = {}) {
  const jsonData = await getModelTextFile(modelPath, fileName, fatal, options)

  return JSON.parse(jsonData)
}
