import Box from '@mui/material/Box'
import List from '@mui/material/List'
import ListItem from '@mui/material/ListItem'
import React from 'react'
import ListItemText from '@mui/material/ListItemText'
import Divider from '@mui/material/Divider'


function FaqItem (props: { question: string, answer:string }){
  return (
    <>
      <ListItem>
        <ListItemText primary={'Q: ' + props.question} />
      </ListItem>
      <ListItem>
        <ListItemText primary={'A: ' + props.answer} />
      </ListItem>
      <Divider/>
    </>
  )
}

export function FAQ () {
  return (
    <Box sx={{ width: '100%', bgcolor: 'background.paper' }}>
      <List>
        <ListItem>
          <h2>FAQ</h2>
        </ListItem>
        <FaqItem question={'What if I get protobuf parsing failed error?'} answer={'Open DevTools, go to Application -> Storage and press "Clear site data".'} />
        <FaqItem question={'What if I get sbox_fatal_memory_exceeded?'} answer={"You don't have enough RAM to run SD. You can try reloading the tab or browser."} />
        <FaqItem question={'How did you make it possible?'} answer={'In order to run it, I had to port StableDiffusionPipeline from python to JS. Then patch onnxruntime and emscripten+binaryen (WebAssembly compiler toolchain) to support allocating and using >4GB memory. Then WebAssembly spec and V8 engine.'} />
      </List>
    </Box>
  )
}
