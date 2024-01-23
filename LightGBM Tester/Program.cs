using System;
using System.IO;
using System.Globalization;
using System.Diagnostics;
using System.Drawing;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Input camera color for training(Red or Black)");
        string cameraColor = Console.ReadLine().ToUpper();


        MLContext mlContext = new MLContext();
        List<PixelData> trainingPixels = new List<PixelData>();
        Console.WriteLine("Paste the path to the Training data File");
        string trainingPath = Console.ReadLine();
        trainingPath = trainingPath.Trim('"');

        // Training Data
        using (StreamReader reader = new StreamReader(trainingPath))
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

                Console.WriteLine("Invalid Path" + ex.Message);
            }
        }

        IDataView trainingData = mlContext.Data.LoadFromEnumerable(trainingPixels);
        Console.WriteLine("Paste path to testing file");
        string testPath = Console.ReadLine();
        testPath = testPath.Trim('"');
        //Testing Data
        List<PixelData> testingPixels = new List<PixelData>();
        using (StreamReader reader = new StreamReader(testPath))
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
        Console.WriteLine("Training...");

        // this is the old LighGBM Model, did not work when implementing into CuDDI(versioning issues)

        //var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Hue", "Saturation", "Intensity" })
        //    .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(PixelData.Color)))
        //    .Append(mlContext.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options()
        //    {
        //        NumberOfLeaves = 31,
        //        MinimumExampleCountPerLeaf = 21,
        //        NumberOfIterations = 140,
        //        LearningRate = 0.047,
        //        UseSoftmax = true,
        //        LabelColumnName = "Label",
        //        FeatureColumnName = "Features"
        //    }))
        //    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Hue", "Saturation", "Intensity" })
        .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(PixelData.Color)))
        .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
            mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features")
            ))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Fast Tree with custom parameters didnt really affect accuracy

        //var pipeline = mlContext.Transforms.Concatenate("Features", "Hue", "Saturation", "Intensity")
        //.Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(PixelData.Color)))
        //.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
        //mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options
        //{
        //    NumberOfLeaves = 31, // Example: maximum leaves in each tree
        //    MinimumExampleCountPerLeaf = 21, // Minimum number of examples per leaf
        //    NumberOfTrees = 140, // Number of trees in the ensemble
        //    LearningRate = 0.047, // Learning rate
        //    LabelColumnName = "Label",
        //    FeatureColumnName = "Features"
        //})
        //))
        //.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));




        var model = pipeline.Fit(trainingData);
        Console.WriteLine("Model Trained");


        Console.WriteLine("Testing...");

        var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testingData));

        Console.WriteLine($"Log-Loss: {testMetrics.LogLoss}");
        Console.WriteLine($"Per class Log-Loss: {string.Join(" , ", testMetrics.PerClassLogLoss.Select(c => c.ToString()))}");
        Console.WriteLine($"Macro Accuracy: {testMetrics.MacroAccuracy}");
        Console.WriteLine($"Micro Accuracy: {testMetrics.MicroAccuracy}");
        Console.WriteLine($"Confusion Matrix:\n {testMetrics.ConfusionMatrix.GetFormattedConfusionTable()}\n");

        Console.WriteLine("Save the Model?");
        var response = Console.ReadLine()?.ToLower();
        if (response == "yes")
        {
            string folderPath = @"C:\Users\rober\OneDrive\Desktop\CuDDI_Models";
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }
            Console.WriteLine("What version is this model?");
            var version = Console.ReadLine();
            mlContext.Model.Save(model, trainingData.Schema, $@"C:\Users\rober\OneDrive\Desktop\CuDDI_Models\FastTreeModel{cameraColor}v{version}_acc_{testMetrics.MacroAccuracy}.zip");
            Console.WriteLine(@$"Model saved to C:\Users\rober\OneDrive\Desktop\CuDDI_Models\FastTreeModel{cameraColor}v{version}_acc_{testMetrics.MacroAccuracy}.zip");
            string metricsPath = $@"C:\Users\rober\OneDrive\Desktop\CuDDI_Models\FastTreeModel{cameraColor}v{version}_acc_{testMetrics.MacroAccuracy}.txt";
            using (var writer = new StreamWriter(metricsPath))
            {
                writer.WriteLine("Metric,Value");
                writer.WriteLine($"Macro Accuracy,{testMetrics.MacroAccuracy.ToString(CultureInfo.InvariantCulture)}");
                writer.WriteLine($"Micro Accuracy,{testMetrics.MicroAccuracy.ToString(CultureInfo.InvariantCulture)}");
                writer.WriteLine($"Log-Loss,{testMetrics.LogLoss.ToString(CultureInfo.InvariantCulture)}");

                // Write per class log-loss
                for (int i = 0; i < testMetrics.PerClassLogLoss.Count; i++)
                {
                    writer.WriteLine($"Class {i} Log-Loss,{testMetrics.PerClassLogLoss[i].ToString(CultureInfo.InvariantCulture)}");
                }
                writer.WriteLine($"Confusion Matrix:\n {testMetrics.ConfusionMatrix.GetFormattedConfusionTable()}\n");
            }
            Console.WriteLine($"Metrics saved to {metricsPath}");
        }
    }
}

public class PixelData
{
    [LoadColumn(0)]
    public float Hue { get; set; }
    [LoadColumn(1)]
    public float Saturation { get; set; }
    [LoadColumn(2)]
    public float Intensity { get; set; }
    [LoadColumn(3)]
    public string Color { get; set; }

}

//public class PixelDataWithLabel 
//{
//    public float Hue { get; set; }
//    public float Saturation { get; set; }
//    public float Intensity { get; set; }
//    public string Color { get; set; }
//}

public class Predicition
{
    [ColumnName("PredictedLabel")]
    public string Color { get; set; }
}



