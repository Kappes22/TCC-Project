package com.example.livefeedtcc

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.Manifest
import android.R.attr
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import kotlinx.android.synthetic.main.activity_main.*
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.ByteArrayOutputStream
import androidx.camera.core.CameraSelector
import android.graphics.Bitmap
import android.graphics.RectF
import android.graphics.BitmapFactory
import android.util.AttributeSet
import android.util.Size
import android.view.View
import android.R.attr.right
import android.graphics.Canvas
import android.graphics.Paint
import android.R.attr.left
import android.util.DisplayMetrics


typealias EmotionListener = (text: String) -> Unit


class MainActivity : AppCompatActivity() {
    private var imageCapture: ImageCapture? = null
    private lateinit var outputDirectory: File
    private lateinit var cameraExecutor: ExecutorService
    var faceDetection = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        textView.text = initPython()
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Set up the listener for take photo button
        camera_capture_button.setOnClickListener { takePhoto() }
        btnFaceDetection.setOnClickListener {
            faceDetection = faceDetection != true
            val python = Python.getInstance()
            val pythonFile = python.getModule("Predictor")
            pythonFile.callAttr  ("setFaceDetection",faceDetection)
        }
        btnFlipCamera.setOnClickListener {
            if(defaultCameraFacing == CameraSelector.DEFAULT_FRONT_CAMERA){
                defaultCameraFacing = CameraSelector.DEFAULT_BACK_CAMERA
            }else{
                defaultCameraFacing = CameraSelector.DEFAULT_FRONT_CAMERA
            }
            startCamera()
        }
        outputDirectory = getOutputDirectory()

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time-stamped output file to hold the image
        val photoFile = File(
            outputDirectory,
            SimpleDateFormat(FILENAME_FORMAT, Locale.US
            ).format(System.currentTimeMillis()) + ".jpg")

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions, ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = Uri.fromFile(photoFile)
                    val msg = "Photo capture succeeded: $savedUri"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)
                }
            })
    }


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() } }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()

                .build()

            val imageAnalyzer = ImageAnalysis.Builder().setTargetResolution(Size(160, 120))
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, EmotionAnalyzer { text ->
                        textView.text = text
                        rect_overlay.invalidate()

                    })
                }


            // Select back camera as a default
            val cameraSelector = defaultCameraFacing

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }


    private class EmotionAnalyzer(private val listener: EmotionListener) : ImageAnalysis.Analyzer {
        var constantTime = System.currentTimeMillis()
        lateinit var imageByte : ByteArray
        var early = 3
        var text = ""

        override fun analyze(image: ImageProxy) {
            var currentTime = System.currentTimeMillis()
            if(currentTime-constantTime>0) {

                //var bitmap = image.toBitmap()
                val bitmap = image.toBitmap()//BitMapToString(bitmap)
                val bitmap2 = if(defaultCameraFacing == CameraSelector.DEFAULT_FRONT_CAMERA) bitmap.rotate(-90f) else bitmap.rotate(90f).flipHor()

                //val bitmap2 = if (bitmap.width > bitmap.height) bitmap.rotate(-90f) else bitmap
                imageByte = BitMapToString(bitmap2)
                text = pythonScript()+pythonGetFaceLocationScript()//pythonScript()//imageString//pythonScript()
                listener(text)
                constantTime = currentTime
            }
            image.close()
        }

        fun Bitmap.rotate(degrees: Float): Bitmap {
            val matrix = Matrix().apply { postRotate(degrees) }
            return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
        }

        fun Bitmap.flipHor(): Bitmap {
            val x = -1f
            val y = 1f
            val cx = width / 2f
            val cy = height / 2f
            val matrix = Matrix().apply { postScale(x, y, cx, cy) }
            return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
        }

        private fun pythonScript():String{
            val python = Python.getInstance()
            val pythonFile = python.getModule("Predictor")
            return pythonFile.callAttr("main", imageByte, early).toString()
        }

        private fun pythonGetFaceLocationScript():String{
            val python = Python.getInstance()
            val pythonFile = python.getModule("Predictor")
            val x = pythonFile.callAttr("getxFace").toFloat()*(screenWidth/ imageWidth)
            val y = pythonFile.callAttr("getyFace").toFloat()*(screenHeight/ imageHeight)
            val w = pythonFile.callAttr("getwFace").toFloat()*(screenWidth/ imageWidth)
            val h = pythonFile.callAttr("gethFace").toFloat()*(screenHeight/ imageHeight)
            rectBounds = RectF((screenWidth - x).toFloat(),y.toFloat(),(screenWidth-x-w).toFloat(),y.toFloat()+h.toFloat())

            return x.toString()+"x "+y.toString()+"y "+w.toString()+"w "+h.toString()+"h "
        }

        fun ImageProxy.toBitmap(): Bitmap {
            val nv21 = yuv420888ToNv21(this)
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            return yuvImage.toBitmap()
        }

        private fun YuvImage.toBitmap(): Bitmap {
            val out = ByteArrayOutputStream()
            compressToJpeg(Rect(0, 0, width, height), 100, out)
            imageWidth = height.toFloat()
            imageHeight = width.toFloat()
            val imageBytes: ByteArray = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }

        private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
            val pixelCount = image.cropRect.width() * image.cropRect.height()
            val pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888)
            val outputBuffer = ByteArray(pixelCount * pixelSizeBits / 8)
            imageToByteBuffer(image, outputBuffer, pixelCount)
            return outputBuffer
        }

        private fun imageToByteBuffer(image: ImageProxy, outputBuffer: ByteArray, pixelCount: Int) {
            assert(image.format == ImageFormat.YUV_420_888)

            val imageCrop = image.cropRect
            val imagePlanes = image.planes

            imagePlanes.forEachIndexed { planeIndex, plane ->
                // How many values are read in input for each output value written
                // Only the Y plane has a value for every pixel, U and V have half the resolution i.e.
                //
                // Y Plane            U Plane    V Plane
                // ===============    =======    =======
                // Y Y Y Y Y Y Y Y    U U U U    V V V V
                // Y Y Y Y Y Y Y Y    U U U U    V V V V
                // Y Y Y Y Y Y Y Y    U U U U    V V V V
                // Y Y Y Y Y Y Y Y    U U U U    V V V V
                // Y Y Y Y Y Y Y Y
                // Y Y Y Y Y Y Y Y
                // Y Y Y Y Y Y Y Y
                val outputStride: Int

                // The index in the output buffer the next value will be written at
                // For Y it's zero, for U and V we start at the end of Y and interleave them i.e.
                //
                // First chunk        Second chunk
                // ===============    ===============
                // Y Y Y Y Y Y Y Y    V U V U V U V U
                // Y Y Y Y Y Y Y Y    V U V U V U V U
                // Y Y Y Y Y Y Y Y    V U V U V U V U
                // Y Y Y Y Y Y Y Y    V U V U V U V U
                // Y Y Y Y Y Y Y Y
                // Y Y Y Y Y Y Y Y
                // Y Y Y Y Y Y Y Y
                var outputOffset: Int

                when (planeIndex) {
                    0 -> {
                        outputStride = 1
                        outputOffset = 0
                    }
                    1 -> {
                        outputStride = 2
                        // For NV21 format, U is in odd-numbered indices
                        outputOffset = pixelCount + 1
                    }
                    2 -> {
                        outputStride = 2
                        // For NV21 format, V is in even-numbered indices
                        outputOffset = pixelCount
                    }
                    else -> {
                        // Image contains more than 3 planes, something strange is going on
                        return@forEachIndexed
                    }
                }

                val planeBuffer = plane.buffer
                val rowStride = plane.rowStride
                val pixelStride = plane.pixelStride

                // We have to divide the width and height by two if it's not the Y plane
                val planeCrop = if (planeIndex == 0) {
                    imageCrop
                } else {
                    Rect(
                        imageCrop.left / 2,
                        imageCrop.top / 2,
                        imageCrop.right / 2,
                        imageCrop.bottom / 2
                    )
                }

                val planeWidth = planeCrop.width()
                val planeHeight = planeCrop.height()

                // Intermediate buffer used to store the bytes of each row
                val rowBuffer = ByteArray(plane.rowStride)

                // Size of each row in bytes
                val rowLength = if (pixelStride == 1 && outputStride == 1) {
                    planeWidth
                } else {
                    // Take into account that the stride may include data from pixels other than this
                    // particular plane and row, and that could be between pixels and not after every
                    // pixel:
                    //
                    // |---- Pixel stride ----|                    Row ends here --> |
                    // | Pixel 1 | Other Data | Pixel 2 | Other Data | ... | Pixel N |
                    //
                    // We need to get (N-1) * (pixel stride bytes) per row + 1 byte for the last pixel
                    (planeWidth - 1) * pixelStride + 1
                }

                for (row in 0 until planeHeight) {
                    // Move buffer position to the beginning of this row
                    planeBuffer.position(
                        (row + planeCrop.top) * rowStride + planeCrop.left * pixelStride)

                    if (pixelStride == 1 && outputStride == 1) {
                        // When there is a single stride value for pixel and output, we can just copy
                        // the entire row in a single step
                        planeBuffer.get(outputBuffer, outputOffset, rowLength)
                        outputOffset += rowLength
                    } else {
                        // When either pixel or output have a stride > 1 we must copy pixel by pixel
                        planeBuffer.get(rowBuffer, 0, rowLength)
                        for (col in 0 until planeWidth) {
                            outputBuffer[outputOffset] = rowBuffer[col * pixelStride]
                            outputOffset += outputStride
                        }
                    }
                }
            }
        }


        fun BitMapToString(bitmap: Bitmap): ByteArray {
            val baos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, baos)
            val b = baos.toByteArray()
            return b
        }
    }

    private fun initPython():String {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        val python = Python.getInstance()
        val pythonFile = python.getModule("helloworldscript")
        return pythonFile.callAttr  ("init").toString()
    }

    class RectOverlay constructor(context: Context?, attributeSet: AttributeSet?) :
        View(context, attributeSet) {

        private val paint = Paint().apply {
            style = Paint.Style.STROKE
            color = ContextCompat.getColor(context!!, android.R.color.black)
            strokeWidth = 10f
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            // Pass it a list of RectF (rectBounds)
            canvas.drawRect(rectBounds, paint );
        }
    }



    companion object {
        private const val TAG = "CameraXBasic"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        val screenWidth = 720F//1080F//
        val screenHeight = 1350F//1974F//
        var imageWidth = 1F
        var imageHeight = 1F
        var rectBounds: RectF = RectF(
            50F,
            50F,
            100F,
            100F
        )
        var defaultCameraFacing = CameraSelector.DEFAULT_FRONT_CAMERA

    }
}
