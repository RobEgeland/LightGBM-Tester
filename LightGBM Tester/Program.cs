using System;
using System.IO;
using System.Globalization;
using System.Diagnostics;
using System.Drawing;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Input camera color for training (Red or Black)");
        string cameraColor = Console.ReadLine().ToUpper();

        MLContext mlContext = new MLContext();
        List<PixelData> pixels = new List<PixelData>();
        Console.WriteLine("Paste the path to the data file");
        string dataPath = Console.ReadLine();
        dataPath = dataPath.Trim('"');

        // Load Data
        using (StreamReader reader = new StreamReader(dataPath))
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

                    pixels.Add(new PixelData()
                    {
                        Hue = hue,
                        Saturation = saturation,
                        Intensity = intensity,
                        Color = values[8]
                    });
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Invalid Path: " + ex.Message);
            }
        }

        // Undersample the data to ensure equal representation of each class
        //pixels = Undersample(pixels);

        // Add Weights here 

        IDataView data = mlContext.Data.LoadFromEnumerable(pixels);

        // Split the data into training and testing sets
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.3);

        IDataView trainingData = split.TrainSet;
        IDataView testingData = split.TestSet;

        Console.WriteLine("Training...");

        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(PixelData.Color))
            .Append(mlContext.Transforms.Concatenate("Features", "Hue", "Saturation", "Intensity"))
            .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
                mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features")))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

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
            string folderPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "CuDDI_Models");
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }
            Console.WriteLine("What version is this model?");
            var version = Console.ReadLine();
            string modelPath = Path.Combine(folderPath, $"FastTreeModel{cameraColor}v{version}_acc_{testMetrics.MacroAccuracy}.zip");
            mlContext.Model.Save(model, trainingData.Schema, modelPath);
            Console.WriteLine(@$"Model saved to {modelPath}");
            string metricsPath = Path.Combine(folderPath, $"FastTreeModel{cameraColor}v{version}_acc_{testMetrics.MacroAccuracy}.txt");
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
                writer.WriteLine($"Model Data: {dataPath}");
            }
            Console.WriteLine($"Metrics saved to {metricsPath}");
        }
    }

    public static List<PixelData> Undersample(List<PixelData> data)
    {
        var groupedData = data.GroupBy(p => p.Color);
        List<PixelData> undersampledData = new List<PixelData>();

        Random rand = new Random();

        foreach (var group in groupedData)
        {
            // Calculate the average values for Hue, Saturation, and Intensity in the group
            double avgHue = group.Average(p => p.Hue);
            double avgSaturation = group.Average(p => p.Saturation);
            double avgIntensity = group.Average(p => p.Intensity);

            // Remove 10% from top and bottom based on deviation from the average
            var scoredGroup = group.Select(p => new
            {
                Pixel = p,
                Deviation = Math.Abs(p.Hue - avgHue) + Math.Abs(p.Saturation - avgSaturation) + Math.Abs(p.Intensity - avgIntensity)
            }).OrderBy(p => p.Deviation).ToList();

            int count = scoredGroup.Count;
            int removeCount = (int)(count * 0.1); // Remove 10% from top and bottom

            // Take the middle portion to maintain average values
            var middlePortion = scoredGroup.Skip(removeCount).Take(count - 2 * removeCount).Select(p => p.Pixel);
            undersampledData.AddRange(middlePortion);
        }

        return undersampledData;
    }



    //public static List<PixelData> Undersample(List<PixelData> data)
    //{
    //    var groupedData = data.GroupBy(p => p.Color);
    //    int minCount = groupedData.Min(g => g.Count());

    //    List<PixelData> undersampledData = new List<PixelData>();
    //    foreach (var group in groupedData)
    //    {
    //        undersampledData.AddRange(group.Take(minCount));
    //    }

    //    return undersampledData;
    //}
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

public class Prediction
{
    [ColumnName("PredictedLabel")]
    public string Color { get; set; }
}




