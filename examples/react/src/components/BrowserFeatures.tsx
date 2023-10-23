import React, { useEffect, useState } from 'react'
import { memory64 } from 'wasm-feature-detect'
import Stack from '@mui/material/Stack'
import Alert from '@mui/material/Alert'

export const BrowserFeatures = () => {
  const [hasMemory64, setHasMemory64] = useState(false);
  const [hasSharedMemory64, setHasSharedMemory64] = useState(false);
  const [hasJspi, setHasJspi] = useState(false);
  const [hasF16, setHasF16] = useState(false);
  const [hasGpu, setHasGpu] = useState(false);

  useEffect(() => {
    memory64().then(value => setHasMemory64(value))
    // @ts-ignore
    setHasJspi(typeof WebAssembly.Function !== 'undefined')

    try {
      // @ts-ignore
      const mem = new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true, index: 'i64' })
      // @ts-ignore
      setHasSharedMemory64(mem.type().index === 'i64')
    } catch (e) {
      //
    }

    try {
      // @ts-ignore
      navigator.gpu.requestAdapter().then((adapter) => {
        setHasF16(adapter.features.has('shader-f16'))
      })
      setHasGpu(true)
    } catch (e) {
      //
    }

  }, [])

  return (
    <Stack>
      {!hasMemory64 && <Alert severity="error">You need latest Chrome with "Experimental WebAssembly" flag enabled!</Alert>}
      {!hasJspi && <Alert severity="error">You need "Experimental WebAssembly JavaScript Promise Integration (JSPI)" flag enabled!</Alert>}
      {!hasSharedMemory64 && <Alert severity="error">You need Chrome Canary 119 or newer!</Alert>}
      {!hasF16 && <Alert severity="error">You need to run chrome with --enable-dawn-features=allow_unsafe_apis on linux/mac or with --enable-dawn-features=allow_unsafe_apis,use_dxc on windows!</Alert>}
      {!hasGpu && <Alert severity="error">You need a browser with WebGPU support!</Alert>}
    </Stack>
  )
}
