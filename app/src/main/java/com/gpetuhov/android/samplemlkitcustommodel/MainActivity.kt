package com.gpetuhov.android.samplemlkitcustommodel

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.google.firebase.ml.common.modeldownload.FirebaseCloudModelSource
import com.google.firebase.ml.common.modeldownload.FirebaseLocalModelSource
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.custom.*
import com.pawegio.kandroid.toast
import kotlinx.android.synthetic.main.activity_main.*
import java.io.IOException
import java.io.InputStream

// This example uses pre-trained TensorFlow Lite model from Google Codelab:
// https://codelabs.developers.google.com/codelabs/mlkit-android-custom-model/#1

// The model is both hosted in Firebase and included in the app's assets folder

// Don't forget to set Internet permission in the manifest!

// Image is taken from:
// https://medium.com/technogise/enhance-your-skills-and-career-using-the-power-of-open-source-808c1dff7a9c

class MainActivity : AppCompatActivity() {

    private var interpreter: FirebaseModelInterpreter? = null
    private lateinit var inputOutputOptions: FirebaseModelInputOutputOptions

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        configureHostedModelSource()
        configureLocalModelSource()
        createInterpreter()
        specifyInputOutput()

        startButton.setOnClickListener { performInference() }
        clearButton.setOnClickListener { resultTextView.text = "" }
    }

    // Configure a Firebase-hosted model source
    private fun configureHostedModelSource() {
        var conditionsBuilder: FirebaseModelDownloadConditions.Builder =
            FirebaseModelDownloadConditions.Builder().requireWifi()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            // Enable advanced conditions on Android Nougat and newer.
            conditionsBuilder = conditionsBuilder
                .requireCharging()
                .requireDeviceIdle()
        }

        val conditions = conditionsBuilder.build()

        // Build a FirebaseCloudModelSource object by specifying the name you assigned the model
        // when you uploaded it in the Firebase console.
        val cloudSource = FirebaseCloudModelSource.Builder("mobilenet_v1_224_quant")
            .enableModelUpdates(true)
            .setInitialDownloadConditions(conditions)
            .setUpdatesDownloadConditions(conditions)
            .build()

        FirebaseModelManager.getInstance().registerCloudModelSource(cloudSource)
    }

    // Configure a local model source
    private fun configureLocalModelSource() {
        val localSource = FirebaseLocalModelSource.Builder("my_local_model") // Assign a name to this model
            .setAssetFilePath("mobilenet_v1_1.0_224_quant.tflite")
            .build()

        FirebaseModelManager.getInstance().registerLocalModelSource(localSource)
    }

    // Create an interpreter from the model sources
    private fun createInterpreter() {
        val options = FirebaseModelOptions.Builder()
            .setCloudModelName("mobilenet_v1_224_quant")
            .setLocalModelName("my_local_model")
            .build()

        interpreter = FirebaseModelInterpreter.getInstance(options)
    }

    // We must know input and output parameters of the .tflite model.
    // In case we don't, follow the instructions at: https://firebase.google.com/docs/ml-kit/android/use-custom-models
    private fun specifyInputOutput() {
        inputOutputOptions = FirebaseModelInputOutputOptions.Builder()
            .setInputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1, 224, 224, 3))
            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1, 5))
            .build()
    }

    private fun getBitmapFromAsset(filePath: String): Bitmap? {
        val inputStream: InputStream
        var bitmap: Bitmap? = null

        try {
            inputStream = assets.open(filePath)
            bitmap = BitmapFactory.decodeStream(inputStream)
        } catch (e: IOException) {
            toast("Error opening image file")
        }

        return bitmap
    }

    private fun performInference() {
        val bitmapFromAsset = getBitmapFromAsset("photo.jpeg")

        if (bitmapFromAsset != null) {
            val bitmap = Bitmap.createScaledBitmap(bitmapFromAsset, 224, 224, true)

            val batchNum = 0
            val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }
            for (x in 0..223) {
                for (y in 0..223) {
                    val pixel = bitmap.getPixel(x, y)
                    // Normalize channel values to [-1.0, 1.0]. This requirement varies by
                    // model. For example, some models might require values to be normalized
                    // to the range [0.0, 1.0] instead.
                    input[batchNum][x][y][0] = (Color.red(pixel) - 127) / 255.0f
                    input[batchNum][x][y][1] = (Color.green(pixel) - 127) / 255.0f
                    input[batchNum][x][y][2] = (Color.blue(pixel) - 127) / 255.0f
                }
            }

            val inputs = FirebaseModelInputs.Builder()
                .add(input) // add() as many input arrays as your model requires
                .build()

            interpreter?.run(inputs, inputOutputOptions)
                ?.addOnSuccessListener { result ->
                    val output = result.getOutput<Array<FloatArray>>(0)
                    val probabilities = output[0]
                }
                ?.addOnFailureListener { exception ->
                    exception.printStackTrace()
                }
        }
    }
}
