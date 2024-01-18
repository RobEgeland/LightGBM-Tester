using System;
using System.Diagnostics;
using System.Drawing;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
class Program
{
    static void Main(string[] args)
    {

        MLContext mlContext = new MLContext();
        List<PixelData> trainingPixels = new List<PixelData>();

        // Training Data
        using (StreamReader reader = new StreamReader(@"C:\Users\rober\OneDrive\Desktop\merged_red_train_10_synthetic.csv"))
        {
          
            string header = reader.ReadLine();
            try
            {
                while (!reader.EndOfStream)
                {
                    
                    string line = reader.ReadLine();
                    if(string.IsNullOrEmpty(line))
                    {
                        continue;
                    }
                    string[] values = line.Split(',');

                    float hue = 0;
                    float saturation = 0;
                    float intensity = 0;

                    // Try parsing, and skip the line if parsing fails
                    if (!float.TryParse(values[0].Trim(), out hue) ||
                        !float.TryParse(values[1].Trim(), out saturation) ||
                        !float.TryParse(values[2].Trim(), out intensity))
                    {
                        continue; // Skip this line if any parse fails
                    }

                    trainingPixels.Add(new PixelData()
                    {
                        Hue = hue,
                        Saturation = saturation,
                        Intensity = intensity,
                        Color = values[3]
                    });

                }
            }
            catch (Exception ex)
            {

                throw ex;
            }
        }
        IDataView trainingData = mlContext.Data.LoadFromEnumerable(trainingPixels);

        //Testing Data
        List<PixelData> testingPixels = new List<PixelData>();
        using (StreamReader reader = new StreamReader(@"C:\Users\rober\OneDrive\Desktop\merged_red_test_2_synthetic.csv"))
        {

            string header = reader.ReadLine();
            try
            {
                while (!reader.EndOfStream)
                {

                    string line = reader.ReadLine();
                    if (string.IsNullOrEmpty(line))
                    {
                        continue;
                    }
                    string[] values = line.Split(',');

                    float hue = 0;
                    float saturation = 0;
                    float intensity = 0;

                    // Try parsing, and skip the line if parsing fails
                    if (!float.TryParse(values[0].Trim(), out hue) ||
                        !float.TryParse(values[1].Trim(), out saturation) ||
                        !float.TryParse(values[2].Trim(), out intensity))
                    {
                        continue; // Skip this line if any parse fails
                    }

                    testingPixels.Add(new PixelData()
                    {
                        Hue = hue,
                        Saturation = saturation,
                        Intensity = intensity,
                        Color = values[3]
                    });

                }
            }
            catch (Exception ex)
            {

                throw ex;
            }
        }
        IDataView testingData = mlContext.Data.LoadFromEnumerable(testingPixels);

        var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Hue", "Saturation", "Intensity" })
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Color"))
            .Append(mlContext.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options()
            {
                NumberOfLeaves = 5,
                MinimumExampleCountPerLeaf = 1,
                NumberOfIterations = 100,
                LearningRate = 0.1,
                UseSoftmax = true,
                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            }))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var model = pipeline.Fit(trainingData);

        // supposed to be Transparent Black
        var color = new PixelData()
        {
            Hue = 330f,
            Saturation = 0.09210525453090668f,
            Intensity = 0.2980392277240753f
        };

        var predictionEngine = mlContext.Model.CreatePredictionEngine<PixelData, Predicition>(model).Predict(color);
        Console.WriteLine("Predicited value should be Transparent Black");
        Console.WriteLine($"Predicted Color for Hue: {color.Hue} = {predictionEngine.Color}\n");
        Console.WriteLine("Evaluating...");

        var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testingData));

        Console.WriteLine($"Log-Loss: {testMetrics.LogLoss}");
        Console.WriteLine($"Per class Log-Loss: {string.Join(" , ", testMetrics.PerClassLogLoss.Select(c => c.ToString()))}");
        Console.WriteLine($"Macro Accuracy: {testMetrics.MacroAccuracy}");
        Console.WriteLine($"Micro Accuracy: {testMetrics.MicroAccuracy}");
        Console.WriteLine($"Confusion Matrix:\n {testMetrics.ConfusionMatrix.GetFormattedConfusionTable()}\n");

        Console.WriteLine("Save the Model?");
        var response = Console.ReadLine()?.ToLower(); 
        if(response == "yes")
        {
            mlContext.Model.Save(model, trainingData.Schema, @"C:\Users\rober\OneDrive\Desktop\LightGBMModel.zip");
        }
        Console.WriteLine("Model saved to C:\\Users\\rober\\OneDrive\\Desktop\\LightGBMModel.zip");
    }
}
    
public class PixelData
{
    public float Hue { get; set; }
    public float Saturation { get; set; }
    public float Intensity { get; set; }
    public string Color { get; set; }
}

public class Predicition
{
    [ColumnName("PredictedLabel")]
    public string Color { get; set; }
}



