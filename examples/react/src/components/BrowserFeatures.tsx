import React, { useEffect, useState } from 'react'
import { memory64, jspi } from 'wasm-feature-detect'
import Stack from '@mui/material/Stack'
import Alert from '@mui/material/Alert'

export async function hasFp16 () {
  try {
    // @ts-ignore
    const adapter = await navigator.gpu.requestAdapter()
    return adapter.features.has('shader-f16')
  } catch (e) {
    return false
  }
}

export const BrowserFeatures = () => {
  const [hasMemory64, setHasMemory64] = useState(false);
  const [hasSharedMemory64, setHasSharedMemory64] = useState(false);
  const [hasJspi, setHasJspi] = useState(false);
  const [hasF16, setHasF16] = useState(false);
  const [hasGpu, setHasGpu] = useState(false);

  useEffect(() => {
    memory64().then(value => setHasMemory64(value))
    // @ts-ignore
    jspi().then(value => setHasJspi(value))

    try {
      // @ts-ignore
      const mem = new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true, index: 'i64' })
      // @ts-ignore
      setHasSharedMemory64(mem.type().index === 'i64')
    } catch (e) {
      //
    }

    hasFp16().then(v => {
      setHasF16(v)
      setHasGpu(true)
    })

  }, [])

  return (
    <Stack>
      {!hasMemory64 && <Alert severity="error">You need latest Chrome with "Experimental WebAssembly" flag enabled!</Alert>}
      {!hasJspi && <Alert severity="error">You need "Experimental WebAssembly JavaScript Promise Integration (JSPI)" flag enabled!</Alert>}
      {!hasSharedMemory64 && <Alert severity="error">You need Chrome Canary 119 or newer!</Alert>}
      {!hasF16 && <Alert severity="error">You need Chrome Canary 121 or higher for FP16 support!</Alert>}
      {!hasGpu && <Alert severity="error">You need a browser with WebGPU support!</Alert>}
    </Stack>
  )
}
