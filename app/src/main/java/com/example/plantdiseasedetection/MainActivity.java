package com.example.plantdiseasedetection;

import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

@ExperimentalGetImage public class MainActivity extends AppCompatActivity {

    private final String[] REQUIRED_PERMISSION = new String[]{"android.permission.CAMERA"};
    PreviewView previewView;
    TextView textView;
    Executor executor = Executors.newSingleThreadExecutor();
    Module module;
    List<String> predicted_classes;
    private ListenableFuture<ProcessCameraProvider> cameraProviderListenableFuture;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.result);
        if (!checkPermission()) {
            int REQUEST_CODE_PERMISSION = 101;
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSION, REQUEST_CODE_PERMISSION);
        }

        predicted_classes = loadClasses();
        loadTorchModule();
        cameraProviderListenableFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderListenableFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderListenableFuture.get();
                startCamera(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                //ERROR
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean checkPermission() {
        for (String permission : REQUIRED_PERMISSION) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    void startCamera(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder().setTargetResolution(new Size(224, 224)).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();
        imageAnalysis.setAnalyzer(executor, image -> {
            int rotation = image.getImageInfo().getRotationDegrees();
            analyzeImage(image, rotation);
            image.close();
        });

        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    void loadTorchModule() {
        File modelFile = new File(this.getFilesDir(), "mobile.ptl");
        try {
            if (!modelFile.exists()) {
                InputStream inputStream = getAssets().open("mobile.ptl");
                FileOutputStream outputStream = new FileOutputStream(modelFile);
                byte[] buffer = new byte[2048];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
                inputStream.close();
                outputStream.close();
            }
            module = LiteModuleLoader.load(modelFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    void analyzeImage(ImageProxy imageProxy, int rotation) {
        Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(Objects.requireNonNull(imageProxy.getImage()), rotation, 224, 224, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        String predicted = predicted_classes.get(maxScoreIdx);
        Log.v("Torch", "Detected - " + predicted);

        runOnUiThread(() -> textView.setText(predicted));
    }

    List<String> loadClasses() {
        List<String> classes = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
            String line;
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classes;
    }
}