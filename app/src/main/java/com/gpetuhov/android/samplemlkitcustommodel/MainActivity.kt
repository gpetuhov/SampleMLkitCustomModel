package com.gpetuhov.android.samplemlkitcustommodel

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*

// This example uses pre-trained TensorFlow Lite model from Google Codelab:
// https://codelabs.developers.google.com/codelabs/mlkit-android-custom-model/#1

// The model is both hosted in Firebase and included in the app's assets folder

// Don't forget to set Internet permission in the manifest!

// Image is taken from:
// https://medium.com/technogise/enhance-your-skills-and-career-using-the-power-of-open-source-808c1dff7a9c

class MainActivity : AppCompatActivity() {

    /**
     * Dimensions of inputs.
     */
    private val DIM_BATCH_SIZE = 1
    private val DIM_PIXEL_SIZE = 3
    private val DIM_IMG_SIZE_X = 224
    private val DIM_IMG_SIZE_Y = 224
    /**
     * Labels corresponding to the output of the vision model.
     */
    private var mLabelList: List<String>? = null

    /* Preallocated buffers for storing image data. */
    private val intValues = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)

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
        mLabelList = loadLabelList(this)

        val inputDims = intArrayOf(DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE)
        val outputDims = intArrayOf(DIM_BATCH_SIZE, mLabelList?.size ?: 0)

        inputOutputOptions = FirebaseModelInputOutputOptions.Builder()
            .setInputFormat(0, FirebaseModelDataType.BYTE, inputDims)
            .setOutputFormat(0, FirebaseModelDataType.BYTE, outputDims)
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
            // Create input data.
            val imgData = convertBitmapToByteBuffer(
                bitmapFromAsset,
                bitmapFromAsset.width,
                bitmapFromAsset.height
            )

            val inputs = FirebaseModelInputs.Builder()
                .add(imgData) // add() as many input arrays as your model requires
                .build()

            interpreter?.run(inputs, inputOutputOptions)
                ?.addOnSuccessListener { result ->
                    val labelProbArray = result.getOutput<Array<ByteArray>>(0)
                    if (labelProbArray != null) {
                        val resultArray = labelProbArray[0]

                        var maxValue = resultArray.max() ?: 0
                        var maxValueIndex = resultArray.indexOf(maxValue)

                        var label = mLabelList?.get(maxValueIndex)
                        resultTextView.text = label
                    }
                }
                ?.addOnFailureListener { exception ->
                    exception.printStackTrace()
                }
        }
    }

    /**
     * Reads label list from Assets.
     */
    private fun loadLabelList(activity: Activity): List<String> {
        val labelList = ArrayList<String>()
        try {
            BufferedReader(InputStreamReader(activity.assets.open("labels.txt"))).use { reader ->
                var line = reader.readLine()
                while (line != null) {
                    labelList.add(line)
                    line = reader.readLine()
                }
            }
        } catch (e: IOException) {
            toast("Failed to read label list.")
        }

        return labelList
    }

    /**
     * Writes Image data into a `ByteBuffer`.
     */
    @Synchronized
    private fun convertBitmapToByteBuffer(
        bitmap: Bitmap, width: Int, height: Int
    ): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE
        )
        imgData.order(ByteOrder.nativeOrder())
        val scaledBitmap = Bitmap.createScaledBitmap(
            bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y,
            true
        )
        imgData.rewind()
        scaledBitmap.getPixels(
            intValues, 0, scaledBitmap.width, 0, 0,
            scaledBitmap.width, scaledBitmap.height
        )
        // Convert the image to int points.
        var pixel = 0
        for (i in 0 until DIM_IMG_SIZE_X) {
            for (j in 0 until DIM_IMG_SIZE_Y) {
                val `val` = intValues[pixel++]
                imgData.put((`val` shr 16 and 0xFF).toByte())
                imgData.put((`val` shr 8 and 0xFF).toByte())
                imgData.put((`val` and 0xFF).toByte())
            }
        }
        return imgData
    }
}
