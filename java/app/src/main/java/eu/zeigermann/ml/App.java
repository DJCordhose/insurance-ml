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

import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.IntNdArray;

public class App {

    public static void main(String[] args) throws Exception {
        try (var savedModelBundle = SavedModelBundle.load("/home/olli/insurance-ml/app/classifier",
                "serve")) {

            var input_matrix = NdArrays.ofFloats(Shape.of(1, 2));
            input_matrix.set(NdArrays.vectorOf(48.0f, 100.0f), 0);
            Tensor input_tensor = TFloat32.tensorOf(input_matrix);
            Map<String, Tensor> inputTensorMap = new HashMap<>();
            inputTensorMap.put("input", input_tensor);

            var myFunction = savedModelBundle.function("serving_default");
            Map<String, Tensor> outputTensorMap = myFunction.call(inputTensorMap);
            var prediction = outputTensorMap.get("output");
            
            // overly complicated way to get the prediction, but so far found no better way
            var probas = prediction.asRawTensor().data().asFloats();
            float redProba = probas.getFloat(0);
            float yellowProba = probas.getFloat(1);
            float greenProba = probas.getFloat(2);
            System.out.println(redProba);
            System.out.println(yellowProba);
            System.out.println(greenProba);
        }
    }
}