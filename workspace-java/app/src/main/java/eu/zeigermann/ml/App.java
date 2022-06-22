package eu.zeigermann.ml;

import java.util.Map;
import java.util.HashMap;

import org.tensorflow.ConcreteFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Add;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TFloat32;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.SessionFunction;

import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.IntNdArray;

import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpRequest.BodyPublishers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;

import com.google.gson.Gson;

class ResponseDTO {
    List<List<Double>> predictions;
}

record DataSet(int age, int maxSpeed, int milesPerYear, int clazz) {
}

record Prediction(double percentageRed, double percentageYellow, double percentageGreen) {

    public static int RED = 0;
    public static int YELLOW = 1;
    public static int GREEN = 2;

    public int predictionClass() {
        if (percentageRed > percentageYellow && percentageRed > percentageGreen) {
            return RED;
        } else if (percentageYellow > percentageRed && percentageYellow > percentageGreen) {
            return YELLOW;
        } else {
            return GREEN;
        }
    }

    public static Prediction fromPredictionClass(int predictionClass) {
        if (predictionClass == RED) {
            return new Prediction(1, 0, 0);
        } else if (predictionClass == YELLOW) {
            return new Prediction(0, 1, 0);
        } else {
            return new Prediction(0, 0, 1);
        }
    }

}

public class App {

    private SessionFunction servingFunction;

    public static void main(String[] args) throws Exception {
        var app = new App();
        String csvFilename;
        if (args.length > 0) {
            csvFilename = args[0];
        } else {
            csvFilename = "../../data/insurance-customers-risk-1500.csv";
//            csvFilename = "../../data/insurance-customers-risk-1500-test.csv";
            // csvFilename = "../../data/insurance-customers-risk-1500-shift.csv";
        }
         
        var rawCsv = app.parseCsv(csvFilename);
        var dataSets = app.convertCsv(rawCsv);
        // var reqString = app.composedReqString(dataSets);
        // System.out.println(reqString);
        // System.exit(0);

        // TODO: COMMENT IN ONE OF THOSE OPTIONS

        // OPTION 1: Use hand written rules
        var predictions = app.predictFromRules(dataSets);

        // OPTION 2: use the TensorFlow serving API
//         var predictions = app.predictFromServer(dataSets);

        // OPTION 3: Use call using JavaCPP / JNI
        // TODO: Need to point this to your trained model
        // app.loadModel("/home/olli/insurance-ml/app/classifier");
//         app.loadModel("../../app/classifier");
//         var predictions = app.predictFromLocalModel(dataSets);

        var groundTruth = dataSets;
        var detectedTruth = predictions;
        var score = app.score(groundTruth, detectedTruth);
        System.out.println(score);
    }

    public List<Prediction> predictFromRules(List<DataSet> dataSets) {
        List<Prediction> predictions = new ArrayList<>();
        for (var dataSet : dataSets) {
            Prediction prediction = this.predictFromRule(dataSet);
            predictions.add(prediction);
        }
        return predictions;
    }

    // TODO: YOUR RULES HERE
    public Prediction predictFromRule(DataSet dataSet) {
        var prediction = Prediction.fromPredictionClass(Prediction.GREEN);
        if (dataSet.age() > 70) {
            prediction = Prediction.fromPredictionClass(Prediction.RED);
            return prediction;
        }
        if (dataSet.age() < 35 && dataSet.maxSpeed() < 115) {
            prediction = Prediction.fromPredictionClass(Prediction.YELLOW);
            return prediction;
        }
        if (dataSet.age() < 50 && dataSet.age() > 25 && dataSet.maxSpeed() > 140) {
            prediction = Prediction.fromPredictionClass(Prediction.RED);
            return prediction;
        }
        if (dataSet.age() < 35) {
            prediction = Prediction.fromPredictionClass(Prediction.RED);
            return prediction;
        }
        return prediction;
    }

    public List<Prediction> predictFromLocalModel(List<DataSet> dataSets) {
        List<Prediction> predictions = new ArrayList<>();
        for (var dataSet : dataSets) {
            Prediction prediction = predict((float) dataSet.age(), (float) dataSet.maxSpeed());
            predictions.add(prediction);
        }
        return predictions;
    }

    public double score(List<DataSet> groundTruth, List<Prediction> detectedTruth) {
        var total = groundTruth.size();
        var correct = 0;
        for (var i = 0; i < groundTruth.size(); i++) {
            if (groundTruth.get(i).clazz() == detectedTruth.get(i).predictionClass()) {
                correct++;
            }
        }
        return (double) correct / total;
    }

    public List<DataSet> convertCsv(List<List<String>> rows) {
        var dataSets = new ArrayList<DataSet>();
        for (var row : rows) {
            var dataSet = new DataSet((int) Double.parseDouble(row.get(1)), (int) Double.parseDouble(row.get(0)),
                    (int) Double.parseDouble(row.get(2)), (int) Double.parseDouble(row.get(3)));
            dataSets.add(dataSet);
        }
        return dataSets;
    }

    // https://www.baeldung.com/java-csv-file-array
    public List<List<String>> parseCsv(String csvFilename) throws IOException {
        List<List<String>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csvFilename))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue;
                }
                String[] values = line.split(",");
                records.add(Arrays.asList(values));
            }
            return records;
        }
    }

    private String composedReqString(List<DataSet> dataSets) {
        var sb = new StringBuilder();
        boolean first = true;
        for (var dataSet : dataSets) {
            var age = dataSet.age();
            var maxSpeed = dataSet.maxSpeed();
            if (first) {
                first = false;
            } else {
                sb.append(",");
            }
            var entry = String.format("[%s, %s]", age, maxSpeed);
            sb.append(entry);
        }
        var data = String.format("{\"instances\": [%s]}", sb.toString());
        return data;
    }

    public List<Prediction> predictFromServer(List<DataSet> dataSets) throws IOException, InterruptedException {
        var model = "insurance";
        var version = "v1";
        // var data = "{\"instances\": [[50.0, 122.0], [48, 100], [30, 150]]}";
        var data = this.composedReqString(dataSets);
        // var data = "{ \"speed\": 100, \"age\": 51, \"miles\": 10 }";
        var url = String.format("http://localhost:8501/%s/models/%s:predict", version, model);
        var uri = URI.create(url);
        // var url = "http://localhost:8080/predict";
        // System.out.println(uri);
        var request = HttpRequest.newBuilder()
                .uri(uri)
                .header("Content-Type", "application/json")
                .POST(BodyPublishers.ofString(data))
                // .GET()
                .build();

        var client = HttpClient.newHttpClient();
        var response = client.send(request, HttpResponse.BodyHandlers.ofString());
        // System.out.println(response.statusCode());
        // System.out.println(response.body());

        var responseDTO = new Gson().fromJson(response.body(), ResponseDTO.class);
        // System.out.println(responseDTO.predictions);

        List<Prediction> predictions = new ArrayList<>();
        for (var prediction : responseDTO.predictions) {
            // System.out.println(prediction);
            double percentageRed = prediction.get(0);
            double percentageYellow = prediction.get(1);
            double percentageGreen = prediction.get(2);
            var predictResult = new Prediction(percentageRed, percentageYellow, percentageGreen);
            predictions.add(predictResult);
        }
        return predictions;

    }

    public void loadModel(String path) throws Exception {
        var savedModelBundle = SavedModelBundle.load(path, "serve");
        this.servingFunction = savedModelBundle.function("serving_default");
    }

    public Prediction predict(float age, float speed) {

        var input_matrix = NdArrays.ofFloats(Shape.of(1, 2));
        input_matrix.set(NdArrays.vectorOf(age, speed), 0);
        Tensor input_tensor = TFloat32.tensorOf(input_matrix);
        Map<String, Tensor> inputTensorMap = new HashMap<>();
        inputTensorMap.put("input", input_tensor);

        Map<String, Tensor> outputTensorMap = this.servingFunction.call(inputTensorMap);
        var prediction = outputTensorMap.get("output");
        
        // overly complicated way to get the prediction, but so far found no better way
        var probas = prediction.asRawTensor().data().asFloats();
        float redProba = probas.getFloat(0);
        float yellowProba = probas.getFloat(1);
        float greenProba = probas.getFloat(2);

        return new Prediction(redProba, yellowProba, greenProba);

    }
}
