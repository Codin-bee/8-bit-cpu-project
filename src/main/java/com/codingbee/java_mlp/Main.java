package com.codingbee.java_mlp;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        Dataset data = new Dataset();
        data.loadData("C:\\Users\\theco\\Digits\\small4", 10);
        Dataset testData = new Dataset();
        testData.loadFromJson("C:\\Users\\theco\\Digits\\testSet.json");
        System.out.println("Datasets were loaded");
        MLP simple = new MLP(784, 10, new int[]{20, 10, 10}, "src/main/resources/n1");


        //simple.initializeWithRandomValues();
        //simple.saveToFiles("src/main/resources/n2");
        simple.initializeFromFiles("src/main/resources/n1");

//        System.out.println("Training started: " + LocalDateTime.now());
//        simple.train(data, 20, true);
//        System.out.println("Training finished: " + LocalDateTime.now());

        System.out.println("Training data:");

        System.out.println(simple.calculateAverageCost(data));
        System.out.println(simple.getCorrectPercentage(data));

        System.out.println("Testing data:");

        System.out.println(simple.calculateAverageCost(testData));
        System.out.println(simple.getCorrectPercentage(testData));


//        float smallest = 10034.807f;
//        float actual;
//        for (int i = 0; i < 1000; i++) {
//            actual = simple.calculateAverageCost(data) + simple.calculateAverageCost(testData);
//            if (actual < smallest){
//                simple.initializeWithRandomValues();
//                smallest = actual;
//                simple.saveToFiles("src/main/resources/n2");
//                System.out.println("At iteration " + i + " better cost was found: " + smallest);
//            }else{
//                System.out.println(i + ": " + actual);
//            }
//        }
    }
}
