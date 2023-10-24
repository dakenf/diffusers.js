import { IDBPDatabase, openDB } from 'idb'
import { dispatchProgress, ProgressCallback, ProgressStatus } from '@/pipelines/common'

interface FileMetadata {
  chunks: number;
  chunkLength: number;
  totalLength: number;
  chunk: number;
  file: ArrayBuffer;
}

const DEFAULT_CHUNK_LENGTH = 1024 * 1024 * 512

export class DbCache {
  dbName = 'diffusers-cache'
  dbVersion = 1
  db!: IDBPDatabase

  init = async () => {
    const openRequest = await openDB(this.dbName, this.dbVersion, {
      upgrade (db) {
        if (!db.objectStoreNames.contains('files')) {
          db.createObjectStore('files')
        }
      },
    })

    this.db = openRequest
  }

  storeFile = async (file: ArrayBuffer, name: string, chunkLength = DEFAULT_CHUNK_LENGTH) => {
    const transaction = this.db.transaction(['files'], 'readwrite')
    const store = transaction.objectStore('files')

    const chunks = Math.ceil(file.byteLength / chunkLength)

    const fileMetadata = {
      chunks,
      chunkLength,
      totalLength: file.byteLength,
    }

    for (let i = 0; i < chunks; i++) {
      const chunk = file.slice(i * chunkLength, (i + 1) * chunkLength)
      const nameSuffix = i > 0 ? `-${i}` : ''
      const thisChunkLength = chunk.byteLength
      await store.put({ ...fileMetadata, chunkLength: thisChunkLength, file: chunk, chunk: i }, `${name}${nameSuffix}`)
    }
    await transaction.done
  }

  retrieveFile = async (filename: string, progressCallback: ProgressCallback, displayName: string): Promise<FileMetadata | null> => {
    const transaction = this.db.transaction(['files'], 'readonly')
    const store = transaction.objectStore('files')
    const request = await store.get(filename) as FileMetadata
    if (!request) {
      return null
    }

    if (request.chunks === 1) {
      return request
    }

    let buffer: ArrayBuffer
    if (request.totalLength > 2 * 1024 * 1024 * 1024) {
      // @ts-ignore
      const memory = new WebAssembly.Memory({ initial: Math.ceil(request.totalLength / 65536), index: 'i64' })
      buffer = memory.buffer
    } else {
      buffer = new ArrayBuffer(request.totalLength)
    }

    const baseChunkLength = request.chunkLength
    let view = new Uint8Array(buffer, 0, request.chunkLength)
    view.set(new Uint8Array(request.file))

    await dispatchProgress(progressCallback, {
      status: ProgressStatus.Downloading,
      downloadStatus: {
        file: displayName,
        size: request.totalLength,
        downloaded: request.chunkLength,
      }
    })

    for (let i = 1; i < request.chunks; i++) {
      const file = await store.get(`${filename}-${i}`) as FileMetadata
      view = new Uint8Array(buffer, i * baseChunkLength, file.file.byteLength)
      view.set(new Uint8Array(file.file as ArrayBuffer))

      await dispatchProgress(progressCallback, {
        status: ProgressStatus.Downloading,
        downloadStatus: {
          file: displayName,
          size: request.totalLength,
          downloaded: i * baseChunkLength + file.file.byteLength
        }
      })
    }
    await transaction.done

    return { ...request, file: buffer }
  }
}
