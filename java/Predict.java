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

class DataSet {
    public int age;
    public int maxSpeed;
    public int milesPerYear;
    public int clazz;
}

class Prediction {
    public static int RED = 0;
    public static int YELLOW = 1;
    public static int GREEN = 2;

    public double percentageRed;
    public double percentageYellow;
    public double percentageGreen;

    public Prediction(double percentageRed, double percentageYellow, double percentageGreen) {
        this.percentageRed = percentageRed;
        this.percentageYellow = percentageYellow;
        this.percentageGreen = percentageGreen;
    }

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

public class Predict {

    public static void main(String[] args) throws IOException, InterruptedException {
        var main = new Predict();
        var csvFilename = args[0];
        var rawCsv = main.parseCsv(csvFilename);
        var dataSets = main.convertCsv(rawCsv);
        // var reqString = main.composedReqString(dataSets);
        // System.out.println(reqString);
        // System.exit(0);

        // main.predict(datasets);
        // var predictions = main.predictFromRules(dataSets);
        var predictions = main.predictFromServer(dataSets);
        for (var prediction : predictions) {
            System.out.println(prediction.predictionClass());
        }

        var groundTruth = dataSets;
        var detectedTruth = predictions;
        var score = main.score(groundTruth, detectedTruth);
        System.out.println(score);
    }

    public double score(List<DataSet> groundTruth, List<Prediction> detectedTruth) {
        var total = groundTruth.size();
        var correct = 0;
        for (var i = 0; i < groundTruth.size(); i++) {
            if (groundTruth.get(i).clazz == detectedTruth.get(i).predictionClass()) {
                correct++;
            }
        }
        return (double) correct / total;
    }

    public List<DataSet> convertCsv(List<List<String>> rows) {
        var dataSets = new ArrayList<DataSet>();
        for (var row : rows) {
            var dataSet = new DataSet();
            dataSet.age = (int) Double.parseDouble(row.get(1));
            dataSet.maxSpeed = (int) Double.parseDouble(row.get(0));
            dataSet.milesPerYear = (int) Double.parseDouble(row.get(2));
            dataSet.clazz = (int) Double.parseDouble(row.get(3));
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

    public List<Prediction> predictFromRules(List<DataSet> dataSets) {
        List<Prediction> predictions = new ArrayList<>();
        for (var dataSet : dataSets) {
            Prediction prediction = Prediction.fromPredictionClass(Prediction.GREEN);
            predictions.add(prediction);
        }
        return predictions;
    }

    private String composedReqString(List<DataSet> dataSets) {
        var sb = new StringBuilder();
        boolean first = true;
        for (var dataSet : dataSets) {
            var age = dataSet.age;
            var maxSpeed = dataSet.maxSpeed;
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
        System.out.println(uri);
        var request = HttpRequest.newBuilder()
                .uri(uri)
                .header("Content-Type", "application/json")
                .POST(BodyPublishers.ofString(data))
                // .GET()
                .build();

        var client = HttpClient.newHttpClient();
        var response = client.send(request, HttpResponse.BodyHandlers.ofString());
        System.out.println(response.statusCode());
        System.out.println(response.body());

        var responseDTO = new Gson().fromJson(response.body(), ResponseDTO.class);
        System.out.println(responseDTO.predictions);

        List<Prediction> predictions = new ArrayList<>();
        for (var prediction : responseDTO.predictions) {
            System.out.println(prediction);
            double percentageRed = prediction.get(0);
            double percentageYellow = prediction.get(1);
            double percentageGreen = prediction.get(2);
            var predictResult = new Prediction(percentageRed, percentageYellow, percentageGreen);
            predictions.add(predictResult);
        }
        return predictions;

    }

}