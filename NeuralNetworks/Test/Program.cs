using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;
using NeuralNetwork.Common;

using NeuralNetwork.Layers;
using NeuralNetwork.Common.Activators;

using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using NeuralNetwork.Serialization;
using Newtonsoft.Json;


using System.IO;


using NeuralNetwork.Common.GradientAdjustmentsParameters;
namespace TestPricing
{
    class Program
    {
        static void Main(string[] args)
        {

            int batchSize = 256;
            IActivator activator = new ReLU();
            int numberLayers = 4;
            //Standard( int batchSize, IActivator activator, double[] bias, double[,] weights, IGradientAdjustmentParameters gradientAdjustmentParameters)
            Network network = new Network(batchSize, numberLayers);
            /*network.Layers[0] = new Standard(batchSize, activator, new double[256], new double[7, 256], new FixedLearningRateParameters(0.1));
             network.Layers[1] = new Standard(batchSize, activator, new double[256], new double[256, 256], new FixedLearningRateParameters(0.1));
             network.Layers[2] = new Standard(batchSize, activator, new double[256], new double[256, 256], new FixedLearningRateParameters(0.1));
             network.Layers[3] = new Standard(batchSize, activator, new double[1], new double[1, 1], new FixedLearningRateParameters(0.1));*/

            network.Layers[0] = new Standard(batchSize, activator, new double[200], new double[7, 200], new FixedLearningRateParameters(0.1));
            network.Layers[1] = new Standard(batchSize, activator, new double[130], new double[200, 130], new FixedLearningRateParameters(0.1));
            network.Layers[2] = new Standard(batchSize, activator, new double[50], new double[130, 50], new FixedLearningRateParameters(0.1));
            network.Layers[3] = new Standard(batchSize, activator, new double[1], new double[50, 1], new FixedLearningRateParameters(0.1));

            var serialized = NetworkSerializer.Serialize(network);
             JsonSerializer serializer = new JsonSerializer();
             var filename = "../../../my-network.json";
             using (StreamWriter sw = new StreamWriter(filename))
             using (JsonWriter writer = new JsonTextWriter(sw))
             {
                 serializer.Serialize(writer, serialized);
             }
            Console.Read();
        }
    }
}
