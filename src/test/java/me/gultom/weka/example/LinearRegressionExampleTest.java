package me.gultom.weka.example;

import org.junit.Test;
import weka.core.DenseInstance;

import java.util.Arrays;

public class LinearRegressionExampleTest {

    @Test
    public void test_LinearRegressionExample() {
        LinearRegressionExample example = null;
        try {
            example = new LinearRegressionExample();
            assert example.getDataset().numAttributes() == LinearRegressionExample.DATASET_ATTRIBUTES_NUM;
            assert example.getDataset().numInstances() == LinearRegressionExample.DATASET_SIZE;

            double rmse = example.crossValidate(10);
            System.out.println(String.format("Root squared mean error (RMSE): %e", rmse));
            assert rmse <= 1.0E-7;

            double[] data = new double[]{1.0, 4.0, 316.307, 223.270, 61.543, 175.586, 302.448, 0.0, 65556.0, 44914.0, 188411.0, 14793.0, 539.577};
            double expectation = data[data.length - 1];
            double prediction = example.predict(data);
            System.out.println("Test data: ");
            for (double d : data) {
                System.out.print(d + " ");
            }
            System.out.println();
            System.out.println("Expectation: " + expectation);
            System.out.println("Prediction: " + prediction);
            assert Math.abs(expectation - prediction) <= 1.0E-7;
        } catch (Exception e) {
            e.printStackTrace();
            assert false;
        }
    }

}
