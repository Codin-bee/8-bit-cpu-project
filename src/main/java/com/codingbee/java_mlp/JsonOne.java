package com.codingbee.java_mlp;

@SuppressWarnings("unused")
public class JsonOne {
    private int correctNeuronIndex;
    private int[] values;

    public JsonOne(int correctNeuronIndex, int[] values) {
        this.correctNeuronIndex = correctNeuronIndex;
        this.values = values;
    }

    public JsonOne(){
    }
    public int getCorrectNeuronIndex() {
        return correctNeuronIndex;
    }

    public void setCorrectNeuronIndex(int correctNeuronIndex) {
        this.correctNeuronIndex = correctNeuronIndex;
    }

    public int[] getValues() {
        return values;
    }

    public void setValues(int[] values) {
        this.values = values;
    }
}
