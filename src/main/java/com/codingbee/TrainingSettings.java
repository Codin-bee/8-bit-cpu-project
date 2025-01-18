package com.codingbee;

@SuppressWarnings("unused")
public class TrainingSettings {
    private double learningRate, exponentialDecayRateOne, exponentialDecayRateTwo, epsilon;

    public TrainingSettings(double learningRate, double exponentialDecayRateOne, double exponentialDecayRateTwo, double epsilon) {
        this.learningRate = learningRate;
        this.exponentialDecayRateOne = exponentialDecayRateOne;
        this.exponentialDecayRateTwo = exponentialDecayRateTwo;
        this.epsilon = epsilon;
    }

    public TrainingSettings() {
        this(0.001, 0.9, 0.999, 1e-8);
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setExponentialDecayRateOne(double exponentialDecayRateOne) {
        this.exponentialDecayRateOne = exponentialDecayRateOne;
    }

    public void setExponentialDecayRateTwo(double exponentialDecayRateTwo) {
        this.exponentialDecayRateTwo = exponentialDecayRateTwo;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    //region GETTERS

    public double getLearningRate() {
        return learningRate;
    }

    public double getExponentialDecayRateOne() {
        return exponentialDecayRateOne;
    }

    public double getExponentialDecayRateTwo() {
        return exponentialDecayRateTwo;
    }

    public double getEpsilon() {
        return epsilon;
    }
    //endregion
}
