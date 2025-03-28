/* eslint-disable @typescript-eslint/no-var-requires */
import * as React from 'react'

import { StyleSheet, View, Text, ActivityIndicator } from 'react-native'
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
} from 'react-native-fast-tflite'
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera'
import { useResizePlugin } from 'vision-camera-resize-plugin'

function tensorToString(tensor: Tensor): string {
  return `\n  - ${tensor.dataType} ${tensor.name}[${tensor.shape}]`
}
function modelToString(model: TensorflowModel): string {
  return (
    `TFLite Model (${model.delegate}):\n` +
    `- Inputs: ${model.inputs.map(tensorToString).join('')}\n` +
    `- Outputs: ${model.outputs.map(tensorToString).join('')}`
  )
}

export default function App(): React.ReactNode {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')

  // from https://www.kaggle.com/models/tensorflow/efficientdet/frameworks/tfLite
  const model = useTensorflowModel(require('../assets/hand_landmark_full.tflite'))
  const actualModel = model.state === 'loaded' ? model.model : undefined

  React.useEffect(() => {
    if (actualModel == null) return
    console.log(`Model loaded! Shape:\n${modelToString(actualModel)}]`)
  }, [actualModel])

  const { resize } = useResizePlugin()

  let handDetectedStart = React.useRef<number | null>(null)
  let photoLogged = false

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'

      if (actualModel == null) return

      const resized = resize(frame, {
        scale: {
          width: 224,
          height: 224,
        },
        pixelFormat: 'rgb',
        dataType: 'float32',
      })

      const result = actualModel.runSync([resized])
      const landmarks = result[0]
      const handPresence = result[1]?.[0] ?? 0
      const handedness = result[2]?.[0] ?? 0

      const now = Date.now()

      if (handPresence >= 0.2) {
        if (handDetectedStart.current === null) {
          handDetectedStart.current = now
        } else if (!photoLogged && now - handDetectedStart.current > 1000) {
          console.log('📸 Take picture')
          photoLogged = true
        }

        console.log('Handedness:', handedness > 0.5 ? 'Right' : 'Left')
        console.log("Coord : ", landmarks[0])
      } else {
        handDetectedStart.current = null
        photoLogged = false
        console.log('No hand detected')
      }
    },
    [actualModel]
  )
  

  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])

  console.log(`Model: ${model.state} (${model.model != null})`)

  return (
    <View style={styles.container}>
      {hasPermission && device != null ? (
        <Camera
          device={device}
          style={StyleSheet.absoluteFill}
          isActive={true}
          frameProcessor={frameProcessor}
          pixelFormat="yuv"
        />
      ) : (
        <Text>No Camera available.</Text>
      )}

      {model.state === 'loading' && (
        <ActivityIndicator size="small" color="white" />
      )}

      {model.state === 'error' && (
        <Text>Failed to load model! {model.error.message}</Text>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
})
