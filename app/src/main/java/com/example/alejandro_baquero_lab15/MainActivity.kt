package com.example.alejandro_baquero_lab15   // keep the package line Android Studio generated for you

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private val imgSize = 224
    private val testImages = listOf("test1.png", "test2.png", "test3.png", "test4.png")
    private var current = 0

    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>
    private lateinit var bitmap: Bitmap

    private lateinit var imageView: ImageView
    private lateinit var txtOutput: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageToLabel)
        txtOutput = findViewById(R.id.txtOutput)
        val btnLabel = findViewById<Button>(R.id.btnLabel)
        val btnNext = findViewById<Button>(R.id.btnNext)

        interpreter = Interpreter(loadModelFile("model.tflite"))
        labels = assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }

        showImage()

        btnNext.setOnClickListener {
            current = (current + 1) % testImages.size
            showImage()
        }

        btnLabel.setOnClickListener {
            val probs = classify(bitmap)
            val sb = StringBuilder("${testImages[current]}\n")
            probs.indices.sortedByDescending { probs[it] }.forEach {
                sb.append("${labels[it]} : ${probs[it]}\n")
            }
            txtOutput.text = sb.toString()
        }
    }

    private fun showImage() {
        bitmap = assets.open(testImages[current]).use { BitmapFactory.decodeStream(it) }
        imageView.setImageBitmap(bitmap)
        txtOutput.text = testImages[current]
    }

    private fun loadModelFile(name: String): MappedByteBuffer {
        val fd = assets.openFd(name)
        val channel = FileInputStream(fd.fileDescriptor).channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    // Same preprocessing as Colab: RGB, 224x224, float values 0..255
    // (the model rescales to -1..1 internally)
    private fun classify(src: Bitmap): FloatArray {
        val argb = src.copy(Bitmap.Config.ARGB_8888, false)
        val scaled = Bitmap.createScaledBitmap(argb, imgSize, imgSize, true)

        val input = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(imgSize * imgSize)
        scaled.getPixels(pixels, 0, imgSize, 0, 0, imgSize, imgSize)
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF).toFloat())  // R
            input.putFloat(((p shr 8) and 0xFF).toFloat())   // G
            input.putFloat((p and 0xFF).toFloat())           // B
        }
        input.rewind()

        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)
        return output[0]
    }

    override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }
}
