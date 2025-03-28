/* eslint-disable @typescript-eslint/no-var-requires */
import * as React from 'react';
import {
  StyleSheet,
  View,
  Text,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
} from 'react-native-fast-tflite';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { Worklets } from 'react-native-worklets-core';

// Get screen dimensions
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

const SQUARE_SIZE = Math.min(SCREEN_WIDTH, SCREEN_HEIGHT);

// Helper function to format tensor info (optional, for debugging)
function tensorToString(tensor: Tensor): string {
  return `\n  - ${tensor.dataType} ${tensor.name}[${tensor.shape}]`;
}

// Helper function to format model info (optional, for debugging)
function modelToString(model: TensorflowModel): string {
  return (
    `TFLite Model (${model.delegate}):\n` +
    `- Inputs: ${model.inputs.map(tensorToString).join('')}\n` +
    `- Outputs: ${model.outputs.map(tensorToString).join('')}`
  );
}

export default function App(): React.ReactNode {
  // --- Hooks ---
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back'); // Use 'front' or 'back' as needed
  const model = useTensorflowModel(
    require('../assets/hand_landmark_full.tflite') // Ensure this path is correct
  );
  const { resize } = useResizePlugin();

  // --- Refs ---
  const handDetectedStart = React.useRef<number | null>(null);
  const photoLogged = React.useRef(false); // To prevent repeated logging

  // --- State ---
  const [handednessLabel, setHandednessLabel] =
    React.useState<string>('Detecting...');
  const [landmarkPoints, setLandmarkPoints] = React.useState<
    { x: number; y: number }[]
  >([]);

  // --- Model Loading ---
  const actualModel = model.state === 'loaded' ? model.model : undefined;

  React.useEffect(() => {
    // Log model details once loaded (optional)
    if (actualModel) {
      console.log(`Model loaded! Shape:\n${modelToString(actualModel)}`);
    }
  }, [actualModel]);

  // --- JS Callbacks (to update state from worklet) ---
  const updateHandedness = React.useCallback((label: string) => {
    setHandednessLabel(label);
  }, []);

  const updateLandmarks = React.useCallback((flatLandmarkArray: number[]) => {
    // This function receives the FLAT array from the worklet
    const points: { x: number; y: number }[] = [];
    // The model outputs 21 landmarks, each with x, y, z coordinates (63 values total)
    if (
      !Array.isArray(flatLandmarkArray) ||
      flatLandmarkArray.length === 0
    ) {
      setLandmarkPoints([]); // Clear points if no landmarks
      return;
    }

    for (let i = 0; i < flatLandmarkArray.length; i += 3) {
      // Ensure we have enough elements for x and y
      if (i + 1 < flatLandmarkArray.length) {
        const x = flatLandmarkArray[i];
        const y = flatLandmarkArray[i + 1];
        // const z = flatLandmarkArray[i + 2]; // z is available if needed

        // We only need x and y for 2D drawing
        // Coordinates are relative to the 224x224 input image
        points.push({ x, y });
      }
    }
    setLandmarkPoints(points);
  }, []); // No dependencies needed if it only calls setState

  // --- Worklet Wrappers (to call JS callbacks from worklet) ---
  const updateHandednessJS = Worklets.createRunOnJS(updateHandedness);
  const updateLandmarksJS = Worklets.createRunOnJS(updateLandmarks);

  // --- Frame Processor ---
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'; // IMPORTANT: Marks this function to run on the VisionCamera Frame Processor thread

      if (actualModel == null) {
        // Model isn't loaded yet
        return;
      }

      // 1. Resize the frame for the model input
      //    Input tensor: float32[1, 224, 224, 3]
      const resized = resize(frame, {
        scale: {
          width: 224,
          height: 224,
        },
        pixelFormat: 'rgb', // Model expects RGB
        dataType: 'float32', // Model expects float32
        rotation: "90deg",
      });

      // 2. Run inference
      //    Outputs:
      //    - output_0: float32[1, 63] (landmarks: 21 points * 3 coords [x, y, z])
      //    - output_1: float32[1, 1]  (hand presence score)
      //    - output_2: float32[1, 1]  (handedness score: > 0.5 is Right)
      const result = actualModel.runSync([resized]);

      // 3. Process the results
      const landmarkObject = result[0]; // Landmarks output (Object format: {"0": x0, "1": y0, ...})
      const handPresence = result[1]?.[0] ?? 0; // Hand presence score (default to 0 if undefined)
      const handedness = result[2]?.[0] ?? 0; // Handedness score (default to 0 if undefined)

      const now = Date.now();

      // Check if a hand is likely present
      if (handPresence >= 0.2) {
        // Optional: Hand detection timer logic
        if (handDetectedStart.current === null) {
          handDetectedStart.current = now;
        } else if (!photoLogged.current && now - handDetectedStart.current > 1000) {
          // console.log('📸 Hand detected for 1 second'); // Example action
          photoLogged.current = true; // Prevent repeated logging
        }

        // Determine handedness label
        const label = handedness > 0.5 ? 'Right Hand' : 'Left Hand';
        updateHandednessJS(label); // Update state on JS thread

        // --- FIX: Convert landmark object to a flat array ---
        // Object.values extracts the coordinate values in order
        const flatLandmarkArray = Object.values(landmarkObject) as number[];
        // -----------------------------------------------------

        // Pass the correctly formatted array to the JS thread
        updateLandmarksJS(flatLandmarkArray);

      } else {
        // No hand detected (or low confidence)
        handDetectedStart.current = null; // Reset timer
        photoLogged.current = false;      // Reset log flag
        updateHandednessJS('No Hand Detected'); // Update label
        updateLandmarksJS([]);               // Clear landmarks
      }
    },
    [actualModel, resize, updateHandednessJS, updateLandmarksJS] // Dependencies for the worklet
  );

  // --- Permissions ---
  React.useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // --- Rendering ---
  return (
    <View style={styles.container}>
      {/* Camera View */}
      {device != null && hasPermission ? (
        <Camera
        style={{ width: SQUARE_SIZE, height: SQUARE_SIZE }}
          device={device}
          isActive={true} // Ensure camera is active
          frameProcessor={actualModel ? frameProcessor : undefined} // Only attach processor if model is loaded
          pixelFormat="yuv" // Use yuv for camera efficiency, resize handles conversion
          frameProcessorFps={5} // Optional: Limit FPS to save resources
        />
      ) : (
        <Text style={styles.errorText}>Camera permission required.</Text>
      )}

      {/* Loading Indicator */}
      {model.state === 'loading' && (
        <ActivityIndicator
          style={StyleSheet.absoluteFill}
          size="large"
          color="#FFFFFF"
        />
      )}

      {/* Error Loading Model */}
      {model.state === 'error' && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>
            Failed to load model! {model.error.message}
          </Text>
        </View>
      )}

      {/* Handedness Label */}
      <Text style={styles.handednessText}>{handednessLabel}</Text>

      {/* Landmark Points Overlay */}
      {landmarkPoints.map((pt, idx) => {
        // Check if coordinates are valid numbers before rendering
        if (typeof pt.x !== 'number' || typeof pt.y !== 'number') {
          return null; // Skip rendering if coordinates are invalid
        }

        // Scale landmark coordinates from model input size (224x224) to screen size
        // Subtract half the dot size to center the dot on the landmark
        const displayX = (pt.x / 224) * SQUARE_SIZE - styles.landmarkDot.width / 2;
        const displayY = (pt.y / 224) * SQUARE_SIZE - styles.landmarkDot.height / 2;

        return (
          <View
            key={idx}
            style={[
              styles.landmarkDot,
              {
                left: displayX,
                top: displayY,
              },
            ]}
          />
        );
      })}
    </View>
  );
}

// --- Styles ---
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black', // Background for areas not covered by camera
  },
  errorContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 20,
  },
  errorText: {
    color: 'white', // Changed to white for better visibility on black background
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 10, // Add some spacing if needed
  },
  handednessText: {
    position: 'absolute',
    bottom: 50, // Position from bottom
    left: 20, // Add some padding from edges
    right: 20,
    textAlign: 'center',
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    backgroundColor: 'rgba(0, 0, 0, 0.7)', // Semi-transparent background
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 10, // Rounded corners
    alignSelf: 'center', // Center horizontally relative to left/right bounds
  },
  landmarkDot: {
    position: 'absolute',
    width: 8, // Size of the dot
    height: 8,
    borderRadius: 4, // Make it a circle
    backgroundColor: 'lime', // Bright color for visibility
    // Optional: add a border for contrast
    // borderWidth: 1,
    // borderColor: 'rgba(0, 0, 0, 0.5)',
  },
});