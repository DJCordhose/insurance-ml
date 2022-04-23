import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpRequest.BodyPublishers;
import java.util.ArrayList;
import java.util.List;
import java.io.IOException;
import java.net.URI;

import com.google.gson.Gson;

class ResponseDTO {
  List<List<Double>> predictions;
}

public class Predict {

    public static void main(String[] args) throws IOException, InterruptedException {
        Predict.predict(args);
    }

    public static void predict(String[] args) throws IOException, InterruptedException {
        var model = "insurance";
        var version = "v1";
        var data = "{\"instances\": [[50.0, 122.0], [48, 100], [30, 150]]}";
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
        for (var prediction : responseDTO.predictions) {
            System.out.println(prediction);
            double percentageRed = prediction.get(0);
            double percentageYellow = prediction.get(1);
            double percentageGreen = prediction.get(2);
        }

    }

}