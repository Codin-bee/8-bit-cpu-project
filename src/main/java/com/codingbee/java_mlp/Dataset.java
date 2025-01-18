package com.codingbee.java_mlp;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@SuppressWarnings("unused")
public class Dataset {
    private int[][] inputData;
    public boolean[][] inputs;
    private int[][] expectedResults;


    public Dataset(int[][] inputData, int[][] expectedResults) {
        this.inputData = inputData;
        this.expectedResults = expectedResults;
    }

    public Dataset(){
    }

    public void loadData(String directoryPath, int networkOutputSize){
        int[][] inputData;
        int[][] expectedResults;
        try {
            ObjectMapper mapper = new ObjectMapper();
            File directory = new File(directoryPath);
            File[] files = directory.listFiles((dir, name) -> name.endsWith("json"));
            if (files != null) {
                        inputData = new int[files.length][];
                        expectedResults = new int[files.length][networkOutputSize];
                        for (int i = 0; i < files.length; i++) {
                            Arrays.fill(expectedResults[i], 0);
                            JsonOne example = mapper.readValue(files[i], JsonOne.class);
                            inputData[i] = example.getValues();
                            expectedResults[i][example.getCorrectNeuronIndex()] = 224;
                        }
                this.setInputData(inputData);
                this.setExpectedResults(expectedResults);

                int rows = inputData.length;
                int cols = inputData[0].length;
                inputs = new boolean[rows][cols];

                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        inputs[i][j] = inputData[i][j] == 1;
                    }
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void shuffle(){
        Random rand = new Random();
        int indexOne, indexTwo;
        int[] tempInArray;
        int[] tempOutArray;
        for (int i = 0; i < inputData.length; i++) {
            indexOne = rand.nextInt(inputData.length);
            indexTwo = rand.nextInt(inputData.length );

            tempInArray = inputData[indexOne];
            inputData[indexOne] = inputData[indexTwo];
            inputData[indexTwo] = tempInArray;

            tempOutArray = expectedResults[indexOne];
            expectedResults[indexOne] = expectedResults[indexTwo];
            expectedResults[indexTwo] = tempOutArray;
        }
    }

    public static Dataset mergeDatasets(List<Dataset> datasets){
        List<int[]> inputs = new ArrayList<>();
        List<int[]> outputs = new ArrayList<>();
        for (Dataset dataset : datasets) {
            inputs.addAll(Arrays.asList(dataset.getInputData()));
            outputs.addAll(Arrays.asList(dataset.getExpectedResults()));
        }
        int[][] inputData = inputs.toArray(new int[inputs.size()][]);
        int[][] outputData = outputs.toArray(new int[outputs.size()][]);

        return new Dataset(inputData, outputData);
    }

    public void saveAsJson(String path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File(path), this);
    }

    public void loadFromJson(String path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        Dataset data = mapper.readValue(new File(path), Dataset.class);
        this.expectedResults = data.expectedResults;
        this.inputData = data.inputData;

        int rows = inputData.length;
        int cols = inputData[0].length;
        inputs = new boolean[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                inputs[i][j] = inputData[i][j] == 1;
            }
        }
    }

    public int[][] getInputData() {
        return inputData;
    }

    public void setInputData(int[][] inputData) {
        this.inputData = inputData;
    }

    public int[][] getExpectedResults() {
        return expectedResults;
    }

    public void setExpectedResults(int[][] expectedResults) {
        this.expectedResults = expectedResults;
    }
}