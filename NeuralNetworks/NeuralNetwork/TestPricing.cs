
using NeuralNetwork.Common.Activators;

using NeuralNetwork.Serialization;
using Newtonsoft.Json;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using NeuralNetwork;
using NeuralNetwork.Common;
using NeuralNetwork.Layers;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using NeuralNetwork.Activators;

namespace Pricing
{
    public static class TestPricing
    {
        //Model 1
        static void Main(string[] args)
        {
            int batchSize = 256;
            IActivator activator = new ReLU();
            int numberLayers = 4;
            //Standard( int batchSize, IActivator activator, double[] bias, double[,] weights, IGradientAdjustmentParameters gradientAdjustmentParameters)
            Network network = new Network(batchSize, numberLayers);
           /* network.Layers[0] = new Standard(batchSize, activator, new double[200], new double[200, 200], new FixedLearningRateParameters(0.1));
            network.Layers[1] = new Standard(batchSize, activator, new double[130], new double[130, 130], new FixedLearningRateParameters(0.1));
            network.Layers[2] = new Standard(batchSize, activator, new double[50], new double[50, 50], new FixedLearningRateParameters(0.1));
            network.Layers[3] = new Standard(batchSize, activator, new double[1], new double[1, 1], new FixedLearningRateParameters(0.1));
            var serialized = NetworkSerializer.Serialize(network);
            JsonSerializer serializer = new JsonSerializer();
            var filename = "my-network.json";
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }*/
            Console.ReadKey();
           
        }



    }
}
