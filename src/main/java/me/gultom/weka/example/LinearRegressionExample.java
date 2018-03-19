package me.gultom.weka.example;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * Linear Regression Example
 *
 * Dataset
 *
 * Source: https://archive.ics.uci.edu/ml/datasets/Daily+Demand+Forecasting+Orders
 *
 * Description:
 * The database was collected during 60 days, this is a real database of a Brazilian company of large logistics. Twelve predictive attributes and a target that is the total of orders for daily treatment
 *
 * ARFF header for Weka:
 *
 * @relation Daily_Demand_Forecasting_Orders
 * @attribute Week_of_the_month {1.0, 2.0, 3.0, 4.0, 5.0}
 * @attribute Day_of_the_week_(Monday_to_Friday) {2.0, 3.0, 4.0, 5.0, 6.0}
 * @attribute Non_urgent_order integer
 * @attribute Urgent_order integer
 * @attribute Order_type_A integer
 * @attribute Order_type_B integer
 * @attribute Order_type_C integer
 * @attribute Fiscal_sector_orders integer
 * @attribute Orders_from_the_traffic_controller_sector integer
 * @attribute Banking_orders_(1) integer
 * @attribute Banking_orders_(2) integer
 * @attribute Banking_orders_(3) integer
 * @attribute Target_(Total_orders) integer
 * @data
 *
 */
public class LinearRegressionExample {

    public static String DATASET_FILE = "Daily_Demand_Forecasting_Orders.csv";
    public static int DATASET_SIZE = 60;
    public static int DATASET_ATTRIBUTES_NUM = 13;

    private Instances dataset;
    private LinearRegression model;
    private Normalize normalizer;

    public LinearRegressionExample() throws Exception {
        Instances dataset = loadDataset();
        LinearRegression lr = new LinearRegression();
        lr.setRidge(1.0E-8);

        // normalize data
        Normalize normalizer = new Normalize();
        normalizer.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, normalizer);

        lr.buildClassifier(dataset);

        this.dataset = dataset;
        this.model = lr;
        this.normalizer = normalizer;
    }

    private Instances createEmptyDataset() {
        ArrayList<Attribute> header = this.createHeader();
        Instances instances = new Instances(DATASET_FILE, header, DATASET_SIZE);
        instances.setClassIndex(DATASET_ATTRIBUTES_NUM - 1);
        return instances;
    };

    private Instances loadDataset() throws RuntimeException {
        Instances dataset = null;
        BufferedReader br = null;
        FileReader fr = null;
        try {
            ClassLoader classLoader = getClass().getClassLoader();
            fr = new FileReader(classLoader.getResource(DATASET_FILE).getPath());
            br = new BufferedReader(fr);
            String sCurrentLine;
            int line = 1;

            dataset = this.createEmptyDataset();
            while ((sCurrentLine = br.readLine()) != null) {
                if (line > 1) {
                    try {
                        double[] values = new double[DATASET_ATTRIBUTES_NUM];
                        int i = 0;
                        for (String val : sCurrentLine.split(";")) {
                            values[i] = Double.parseDouble(val);
                            i++;
                        }
                        dataset.add(new DenseInstance(1.0, values));
                    } catch (NumberFormatException ex) {
                        System.err.println(ex.getMessage());
                    }
                }
                line++;
            }
            br.close();
        } catch (final Exception e) {
            throw new RuntimeException(e);
        } finally {
            try {
                if (br != null) br.close();
                if (fr != null) fr.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        return dataset;
    }

    private ArrayList<Attribute> createHeader() {
        ArrayList<Attribute> header = new ArrayList<>();
        header.add(new Attribute("Week_of_the_month"));
        header.add(new Attribute("Day_of_the_week_"));
        header.add(new Attribute("Non_urgent_order"));
        header.add(new Attribute("Urgent_order"));
        header.add(new Attribute("Order_type_A"));
        header.add(new Attribute("Order_type_B"));
        header.add(new Attribute("Order_type_C"));
        header.add(new Attribute("Fiscal_sector_orders"));
        header.add(new Attribute("Orders_from_the_traffic_controller_sector"));
        header.add(new Attribute("Banking_orders_(1)"));
        header.add(new Attribute("Banking_orders_(2)"));
        header.add(new Attribute("Banking_orders_(3)"));
        header.add(new Attribute("Target_(Total_orders)"));
        return header;
    }

    public double crossValidate(int numFolds) throws Exception {
        // cross validate
        Evaluation evaluation = new Evaluation(this.dataset);
        evaluation.crossValidateModel(this.model, dataset, numFolds, new Random(1));
        return evaluation.rootMeanSquaredError();
    }

    public double predict(double[] data) throws Exception {
        Instances instances = this.createEmptyDataset();
        DenseInstance instance = new DenseInstance(1.0, data);
        instances.add(instance);

        instances = Filter.useFilter(instances, this.normalizer);
        System.out.println(instances.instance(0));
        return this.model.classifyInstance(instances.instance(0));
    }

    public Instances getDataset() {
        return dataset;
    }

}
