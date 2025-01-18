package com.codingbee.java_mlp;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@SuppressWarnings("unused")
public class MLP {
    public String networkPath;
    private boolean[][] firstWeights;
    private int[][][] weights;
    private int[][] biases;
    private final int[] hiddenLayersSizes;
    private final int inputLayerSize;
    private final int outputLayerSize;
    public final TrainingSettings trainingSettings = new TrainingSettings();
    public final DebuggingSettings debuggingSettings = new DebuggingSettings();
    private final int LIMIT = 255;

    Random rng = new Random();

    //Parallel processing
    int cores = Runtime.getRuntime().availableProcessors();
    List<boolean[][]> inputBatches = new ArrayList<>();
    List<int[][]> outputBatches = new ArrayList<>();

    public MLP(int inputLayerSize, int outputLayerSize, int[] hiddenLayersSizes, String networkPath){
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.hiddenLayersSizes = hiddenLayersSizes;
        this.networkPath = networkPath;
        allocateWeightMatrices();
    }

    //region Initialization and Saving

    public void initializeFromFiles(String path){
        // Load firstWeights
        File firstWeightsFile = new File(networkPath + File.separator + "firstWeights" + File.separator + "firstWeights.txt");
        if (firstWeightsFile.exists()) {
            List<boolean[]> firstWeightsList = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(firstWeightsFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] values = line.trim().split(" ");
                    boolean[] row = new boolean[values.length];
                    for (int i = 0; i < values.length; i++) {
                        row[i] = values[i].equals("1");
                    }
                    firstWeightsList.add(row);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            firstWeights = firstWeightsList.toArray(new boolean[0][]);
        }

        // Load weights
        File weightsDir = new File(networkPath + File.separator + "weights");
        if (weightsDir.exists() && weightsDir.isDirectory()) {
            File[] weightFiles = weightsDir.listFiles((dir, name) -> name.startsWith("weights_") && name.endsWith(".txt"));
            if (weightFiles != null) {
                weights = new int[weightFiles.length][][];
                for (File weightFile : weightFiles) {
                    int index = Integer.parseInt(weightFile.getName().replace("weights_", "").replace(".txt", ""));
                    List<int[]> weightsList = new ArrayList<>();
                    try (BufferedReader reader = new BufferedReader(new FileReader(weightFile))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            String[] values = line.trim().split(" ");
                            int[] row = new int[values.length];
                            for (int i = 0; i < values.length; i++) {
                                row[i] = Integer.parseInt(values[i]);
                            }
                            weightsList.add(row);
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    weights[index] = weightsList.toArray(new int[0][]);
                }
            }
        }

        // Load biases
        File biasesFile = new File(networkPath + File.separator + "biases" + File.separator + "biases.txt");
        if (biasesFile.exists()) {
            List<int[]> biasesList = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(biasesFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] values = line.trim().split(" ");
                    int[] row = new int[values.length];
                    for (int i = 0; i < values.length; i++) {
                        row[i] = Integer.parseInt(values[i]);
                    }
                    biasesList.add(row);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            biases = biasesList.toArray(new int[0][]);
        }
    }

    public void saveToFiles(String path){
        // Save firstWeights
        File firstWeightsDir = new File(networkPath + File.separator + "firstWeights");
        if (!firstWeightsDir.exists()) {
            firstWeightsDir.mkdirs();
        }
        File firstWeightsFile = new File(firstWeightsDir, "firstWeights.txt");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(firstWeightsFile))) {
            for (boolean[] row : firstWeights) {
                for (boolean value : row) {
                    writer.write(value ? "1" : "0");
                    writer.write(" ");
                }
                writer.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Save weights
        File weightsDir = new File(networkPath + File.separator + "weights");
        if (!weightsDir.exists()) {
            weightsDir.mkdirs();
        }
        for (int i = 0; i < weights.length; i++) {
            File weightFile = new File(weightsDir, "weights_" + i + ".txt");
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(weightFile))) {
                for (int[] row : weights[i]) {
                    for (int value : row) {
                        writer.write(String.valueOf(value));
                        writer.write(" ");
                    }
                    writer.newLine();
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        // Save biases
        File biasesDir = new File(networkPath + File.separator + "biases");
        if (!biasesDir.exists()) {
            biasesDir.mkdirs();
        }
        File biasesFile = new File(biasesDir, "biases.txt");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(biasesFile))) {
            for (int[] row : biases) {
                for (int value : row) {
                    writer.write(String.valueOf(value));
                    writer.write(" ");
                }
                writer.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void initializeWithRandomValues(){
        try {
            for (int i = 0; i < firstWeights.length; i++) {
                for (int j = 0; j < firstWeights[i].length; j++) {
                    firstWeights[i][j] = rng.nextBoolean();
                }
            }
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                        weights[i][j][k] = rng.nextInt(20) - 1;
                    }
                }
            }
            for (int i = 0; i < biases.length; i++) {
                for (int j = 0; j < biases[i].length; j++) {
                    biases[i][j] = rng.nextInt(0, 20);
                }
            }
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }
    //endregion

    //region Processing


    public int[] processAsValues(boolean[] input){
        int[] hiddenLayer1 = new int[hiddenLayersSizes[0]];
        for (int i = 0; i < hiddenLayersSizes[0]; i++) {
            for (int j = 0; j < inputLayerSize; j++) {
                if (input[j] && firstWeights[i][j]) {
                    hiddenLayer1[i] += 2;
                }
                if (hiddenLayer1[i] > 240){
                    break;
                }
            }
            hiddenLayer1[i] -= biases[0][i];
        }
        norm(hiddenLayer1,70, 140);


        int[] hiddenLayer2 = new int[hiddenLayersSizes[1]];
        for (int i = 0; i < hiddenLayersSizes[1]; i++) {
            for (int j = 0; j < hiddenLayersSizes[0]; j++) {
                hiddenLayer2[i] += hiddenLayer1[j] * weights[0][i][j];
                if (hiddenLayer2[i] > 240){
                    break;
                }
            }
            hiddenLayer2[i] -= biases[1][i];
        }
        norm(hiddenLayer2, 35, 50);


        int[] hiddenLayer3 = new int[hiddenLayersSizes[2]];
        for (int i = 0; i < hiddenLayersSizes[2]; i++) {
            for (int j = 0; j < hiddenLayersSizes[1]; j++) {
                hiddenLayer3[i] += hiddenLayer2[j] * weights[1][i][j];
                if (hiddenLayer3[i] > 240){
                    break;
                }
            }
            hiddenLayer3[i] -= biases[2][i];
        }
        norm(hiddenLayer3, 29, 41);


        int[] output = new int[outputLayerSize];
        for (int i = 0; i < outputLayerSize; i++) {
            for (int j = 0; j < hiddenLayersSizes[2]; j++) {
                output[i] += hiddenLayer2[j] * weights[2][i][j];
                if (output[i] > 220){
                    break;
                }
            }
            output[i] -= biases[3][i];
        }

        return output;
    }

    public int processAsIndex(boolean[] input){
        return getIndexWithHighestVal(processAsValues(input));
    }
    //endregion

    //region Training and analyzing

    public void train(Dataset data, int epochs, boolean debugMode){
        setUpParallel(data);
        for (int i = 0; i < epochs; i++) {
            float lower, current, higher, opposite;
            int original;

            //binary weights
            for (int j = 0; j < firstWeights.length; j++) {
                for (int k = 0; k < firstWeights[j].length; k++) {
                    current = calculateAverageCostInParallel(data);
                    firstWeights[j][k] = !firstWeights[j][k];
                    opposite = calculateAverageCostInParallel(data);

                    if (current < opposite){
                        firstWeights[j][k] = !firstWeights[j][k];
                    }
                }
            }
            //weights
            for (int j = 0; j < weights.length; j++) {
                for (int k = 0; k < weights[j].length; k++) {
                    for (int l = 0; l < weights[j][k].length; l++) {
                        original = weights[j][k][l];
                        current = calculateAverageCostInParallel(data);
                        weights[j][k][l] += 1;
                        higher = calculateAverageCostInParallel(data);
                        weights[j][k][l] = Math.max(0, weights[j][k][l] - 2);
                        lower = calculateAverageCostInParallel(data);
                        if (higher < current && higher < lower){
                            weights[j][k][l] = original + 1;
                        }
                        if (current < higher && current < lower){
                            weights[j][k][l] = original;
                        }
                    }
                }
            }
            //biases
            for (int j = 0; j < biases.length; j++) {
                for (int k = 0; k < biases[j].length; k++) {
                    original = biases[j][k];
                    current = calculateAverageCostInParallel(data);
                    biases[j][k] += 1;
                    higher = calculateAverageCostInParallel(data);
                    biases[j][k] -= Math.max(0, biases[j][k] - 2);
                    lower = calculateAverageCostInParallel(data);
                    if (higher < current && higher < lower){
                        biases[j][k] = original + 1;
                    }
                    if (current < higher && current < lower){
                        biases[j][k] = original;
                    }
                }
            }
            System.out.println("Iteration " + i + ": " + calculateAverageCostInParallel(data));
            saveToFiles(networkPath);
            if (i % 2 == 0) System.gc();
        }
    }

    public float calculateAverageCost(Dataset data){
        return calculateAverageCost(data.inputs, data.getExpectedResults());
    }

    float calculateAverageCost(boolean[][] inputData, int[][] expectedOutputData){
        float cost = 0;
        for (int i = 0; i < inputData.length; i++) {
            cost += calculateCost(inputData[i], expectedOutputData[i]);
        }
        return  cost / inputData.length;
    }

    public float calculateCost(boolean[] input, int[] expectedOutput){
        double cost = 0;
        int[] output = processAsValues(input);
        for (int i = 0; i < output.length; i++) {
            double diff = output[i] - expectedOutput[i];
            cost += Math.pow(diff, 2);
        }

        return (float) cost / output.length;
    }

    float calculateAverageCostInParallel(Dataset data){
        return calculateAverageCostInParallel(data.inputs, data.getExpectedResults());
    }

    float calculateAverageCostInParallel(boolean[][] inputs, int[][] expectedOutputs){
        float cost = 0;

        ExecutorService executorService = Executors.newFixedThreadPool(cores);
        List<Future<Float>> futures = new ArrayList<>();

        try {
            for (int i = 0; i < cores; i++) {
                final int index = i;
                futures.add(executorService.submit(() -> calculateAverageCost(
                        inputBatches.get(index),
                        outputBatches.get(index))));
            }

            for (Future<Float> future : futures) {
                cost += future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("An error occurred during parallel computation", e);
        } finally {
            executorService.shutdown();
        }

        return cost / cores;
    }

    public float getCorrectPercentage(Dataset data){
        return getCorrectPercentage(data.inputs, data.getExpectedResults());
    }

    public float getCorrectPercentage(boolean[][] inputData, int[][] expectedOutputData){
        float percentage = 0;
        float count = 0;
        for (int i = 0; i < inputData.length; i++) {
            if (processAsIndex(inputData[i]) == getIndexWithHighestVal(expectedOutputData[i])){
                percentage++;
            }
            count++;
        }
        return ((percentage / count) * 100);
    }
    //endregion

    //region Private Methods
    private void allocateWeightMatrices(){
        firstWeights = new boolean[hiddenLayersSizes[0]][inputLayerSize];
        weights = new int[3][][];
        weights[0] = new int[hiddenLayersSizes[1]][hiddenLayersSizes[0]];
        weights[1] = new int[hiddenLayersSizes[2]][hiddenLayersSizes[1]];
        weights[2] = new int[outputLayerSize][hiddenLayersSizes[2]];
        biases = new int[hiddenLayersSizes.length + 1][];
        biases[biases.length - 1] = new int[outputLayerSize];
        for (int i = 0; i < hiddenLayersSizes.length; i++) {
            biases[i] = new int[hiddenLayersSizes[i]];
        }
    }

    private int getIndexWithHighestVal(int[] array){
        int max = array[0];
        int index = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max){
                max = array[i];
                index = i;
            }
        }
        return index;
    }

    private float calculateWeightGradient(int layer, int to, int from, Dataset data){
        int original = weights[layer][to][from];
        int nudge = 1;
        weights[layer][to][from] = original + nudge;
        float positiveNudge = calculateAverageCost(data);
        weights[layer][to][from] = original - nudge;
        float negativeNudge = calculateAverageCost(data);
        float gradient = (positiveNudge - negativeNudge) / ( 2 * nudge);
        weights[layer][to][from] = original;
        return gradient;
    }

    private float calculateBiasGradient(int layer, int neuron, Dataset data){
        int original = biases[layer][neuron];
        int nudge = 1;
        biases[layer][neuron] = original + nudge;
        float positiveNudge = calculateAverageCost(data);
        biases[layer][neuron] = original - nudge;
        float negativeNudge = calculateAverageCost(data);
        float gradient = (positiveNudge - negativeNudge) / (2 * nudge);
        biases[layer][neuron] = original;
        return gradient;
    }

    private void norm(int[] activations, int t1, int t2){
        for (int i = 0; i < activations.length; i++) {
            if (activations[i] < 0){
                activations[i] = 0;
                continue;
            }
            if (activations[i] <= t1){
                activations[i] = 0;
                continue;
            }
            if (activations[i] > t1 && activations[i] <= t2){
                activations[i] = 1;
                continue;
            }
            if (activations[i] > t2) activations[i] = 2;
        }
    }

    private void setUpParallel(Dataset data){
        int cores = Runtime.getRuntime().availableProcessors();
        System.out.println("Cores used: " + cores);
        inputBatches.clear();
        outputBatches.clear();

        int chunkSize = (int) Math.ceil((double) data.inputs.length / cores);

        for (int i = 0; i < cores; i++) {
            final int startI = i * chunkSize;
            final int endI = Math.min(startI + chunkSize, data.inputs.length);

            if (startI >= endI) {
                break;
            }

            inputBatches.add(Arrays.copyOfRange(data.inputs, startI, endI));
            outputBatches.add(Arrays.copyOfRange(data.getExpectedResults(), startI, endI));
        }
    }

    //endregion
}