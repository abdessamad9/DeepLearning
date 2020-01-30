using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;
using NeuralNetwork.Common;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;

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

            int batchSize = 200;
            IActivator activator = new Identity();// LeakyReLU(); // ReLU();
            int numberLayers = 4;
            Network network = new Network(batchSize, numberLayers);

            /* network.Layers[0] = new Standard(batchSize, activator, new double[200], new double[7, 200], new FixedLearningRateParameters(0.1),1);
             network.Layers[1] = new Standard(batchSize, activator, new double[130], new double[200, 130], new FixedLearningRateParameters(0.1),2);
             network.Layers[2] = new Standard(batchSize, activator, new double[50], new double[130, 50], new FixedLearningRateParameters(0.1),3);
             network.Layers[3] = new Standard(batchSize, activator, new double[1], new double[50, 1], new FixedLearningRateParameters(0.1),4);*/

            /*network.Layers[0] = new Standard(batchSize, activator, new double[200], new double[7, 200], new AdamParameters(0.01, 0.9, 0.99,0.0000001), 1);
            network.Layers[1] = new Standard(batchSize, activator, new double[130], new double[200, 130], new AdamParameters(0.01, 0.9, 0.99, 0.0000001), 2);
            network.Layers[2] = new Standard(batchSize, activator, new double[50], new double[130, 50], new AdamParameters(0.01, 0.9, 0.99, 0.0000001), 3);
            network.Layers[3] = new Standard(batchSize, activator, new double[1], new double[50, 1], new AdamParameters(0.01, 0.9, 0.99, 0.0000001), 4);*/


            network.Layers[0] = new Standard(batchSize, activator, new double[200], new double[7, 200], new MomentumParameters(), 1);
            network.Layers[1] = new Standard(batchSize, activator, new double[130], new double[200, 130], new MomentumParameters(), 2);
            network.Layers[2] = new Standard(batchSize, activator, new double[50], new double[130, 50], new MomentumParameters(), 3);
            network.Layers[3] = new Standard(batchSize, activator, new double[1], new double[50, 1], new MomentumParameters(), 4);

            var serialized = NetworkSerializer.Serialize(network);
             JsonSerializer serializer = new JsonSerializer();
             var filename = "../../../my-network_Identity_Momentom.json";
             using (StreamWriter sw = new StreamWriter(filename))
             using (JsonWriter writer = new JsonTextWriter(sw))
             {
                 serializer.Serialize(writer, serialized);
             }
            Console.Read();
        }
    }
}
