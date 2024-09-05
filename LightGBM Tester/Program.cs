using System;
using System.IO;
using System.Globalization;
using System.Diagnostics;
using System.Drawing;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using System.Collections.Generic;
using System.Linq;

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

        Console.WriteLine("UnderSampling Training Data");

        // Undersample the training data
        trainingPixels = Undersample(trainingPixels);

        IDataView trainingData = mlContext.Data.LoadFromEnumerable(trainingPixels);
        Console.WriteLine("Paste path to testing file");
        string testPath = Console.ReadLine();
        testPath = testPath.Trim('"');

        // Testing Data
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

        // Define hyperparameter options
        int[] numberOfLeavesOptions = { 10, 15, 20, 30 }; // Added more intermediate values to help control complexity
        int[] numberOfTreesOptions = { 50, 100, 150 }; // Increased upper range for larger ensembles if needed
        int[] minExampleCountPerLeafOptions = { 3, 5, 10 }; // Reduced lower bound for more detailed learning if underfitting
        float[] learningRateOptions = { 0.01f, 0.05f, 0.1f, 0.2f }; // Added 0.01 for finer control over convergence
        float[] shrinkageOptions = { 0.7f, 0.8f, 0.9f }; // Added 0.7 for better control over regularization


        double bestAccuracy = 0;
        FastTreeBinaryTrainer.Options bestOptions = null;

        foreach (var leaves in numberOfLeavesOptions)
        {
            foreach (var trees in numberOfTreesOptions)
            {
                foreach (var minExampleCount in minExampleCountPerLeafOptions)
                {
                    foreach (var learningRate in learningRateOptions)
                    {
                        foreach (var shrinkage in shrinkageOptions)
                        {
                            // Define trainer options
                            var options = new FastTreeBinaryTrainer.Options
                            {
                                NumberOfLeaves = leaves,
                                NumberOfTrees = trees,
                                MinimumExampleCountPerLeaf = minExampleCount,
                                LearningRate = learningRate,
                                Shrinkage = shrinkage,
                                LabelColumnName = "Label",
                                FeatureColumnName = "Features"
                            };

                            // Define the pipeline
                            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(PixelData.Color))
                                .Append(mlContext.Transforms.Concatenate("Features", "Hue", "Saturation", "Intensity"))
                                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
                                    mlContext.BinaryClassification.Trainers.FastTree(options)))
                                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                            // Train the model
                            var model = pipeline.Fit(trainingData);

                            // Evaluate the model
                            var predictions = model.Transform(testingData);
                            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

                            // Track the best model
                            if (metrics.MacroAccuracy > bestAccuracy)
                            {
                                bestAccuracy = metrics.MacroAccuracy;
                                bestOptions = options;
                                Console.WriteLine($"New Best Model: Accuracy = {bestAccuracy}, Leaves = {leaves}, Trees = {trees}, LearningRate = {learningRate}, Shrinkage = {shrinkage}");
                            }
                        }
                    }
                }
            }
        }

        // Train the final model with the best hyperparameters
        var finalPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(PixelData.Color))
            .Append(mlContext.Transforms.Concatenate("Features", "Hue", "Saturation", "Intensity"))
            .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
                mlContext.BinaryClassification.Trainers.FastTree(bestOptions)))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var finalModel = finalPipeline.Fit(trainingData);
        Console.WriteLine("Final Model Trained");

        // Evaluate the final model
        var finalMetrics = mlContext.MulticlassClassification.Evaluate(finalModel.Transform(testingData));

        Console.WriteLine($"Final Model Log-Loss: {finalMetrics.LogLoss}");
        Console.WriteLine($"Final Model Per class Log-Loss: {string.Join(" , ", finalMetrics.PerClassLogLoss.Select(c => c.ToString()))}");
        Console.WriteLine($"Final Model Macro Accuracy: {finalMetrics.MacroAccuracy}");
        Console.WriteLine($"Final Model Micro Accuracy: {finalMetrics.MicroAccuracy}");
        Console.WriteLine($"Final Model Confusion Matrix:\n {finalMetrics.ConfusionMatrix.GetFormattedConfusionTable()}\n");

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
            string modelPath = Path.Combine(folderPath, $"FastTreeModel{cameraColor}v{version}_acc_{finalMetrics.MacroAccuracy}.zip");
            mlContext.Model.Save(finalModel, trainingData.Schema, modelPath);
            Console.WriteLine(@$"Model saved to {modelPath}");
            string metricsPath = Path.Combine(folderPath, $"FastTreeModel{cameraColor}v{version}_acc_{finalMetrics.MacroAccuracy}.txt");
            using (var writer = new StreamWriter(metricsPath))
            {
                writer.WriteLine("Metric,Value");
                writer.WriteLine($"Macro Accuracy,{finalMetrics.MacroAccuracy.ToString(CultureInfo.InvariantCulture)}");
                writer.WriteLine($"Micro Accuracy,{finalMetrics.MicroAccuracy.ToString(CultureInfo.InvariantCulture)}");
                writer.WriteLine($"Log-Loss,{finalMetrics.LogLoss.ToString(CultureInfo.InvariantCulture)}");

                // Write per class log-loss
                for (int i = 0; i < finalMetrics.PerClassLogLoss.Count; i++)
                {
                    writer.WriteLine($"Class {i} Log-Loss,{finalMetrics.PerClassLogLoss[i].ToString(CultureInfo.InvariantCulture)}");
                }
                writer.WriteLine($"Confusion Matrix:\n {finalMetrics.ConfusionMatrix.GetFormattedConfusionTable()}\n");
                writer.WriteLine($"Model Training Data: {trainingPath}");
                writer.WriteLine($"Model Testing Data: {testPath}");
            }
            Console.WriteLine($"Metrics saved to {metricsPath}");
        }
    }

    public static List<PixelData> Undersample(List<PixelData> data)
    {
        var groupedData = data.GroupBy(p => p.Color);
        int minCount = groupedData.Min(g => g.Count());

        List<PixelData> undersampledData = new List<PixelData>();
        foreach (var group in groupedData)
        {
            undersampledData.AddRange(group.Take(minCount));
        }

        return undersampledData;
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

public class Prediction
{
    [ColumnName("PredictedLabel")]
    public string Color { get; set; }
}




