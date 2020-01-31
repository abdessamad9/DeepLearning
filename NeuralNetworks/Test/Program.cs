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

        //Renvoie la matrice  gaussienne remplie avec une distribution normale
        public static double[,] rempliMatrice(int nbLignes, int nbColonnes)
        {
            double[,] weights = new double[nbLignes, nbColonnes];
            for (int i = 0; i < nbLignes; i++)
            {
                for (int j = 0; j < nbColonnes; j++)
                {
                    MathNet.Numerics.Distributions.Normal gaussienne = new MathNet.Numerics.Distributions.Normal(0, 1.0 / Math.Sqrt(nbLignes));
                    weights[i, j] = gaussienne.Sample();
                }
            }
            return weights;

        }

        public static void Main(string[] args)
        {

            int batchSize = 200;
            IActivator activator = new Sigmoid(); // Sigmoid();
            int numberLayers = 4;
            Network network = new Network(batchSize, numberLayers);
            MomentumParameters mom = new MomentumParameters();
            mom.LearningRate = 0.1;
            mom.Momentum = 0.5;
            FixedLearningRateParameters fixedL = new FixedLearningRateParameters(0.1);
            AdamParameters adam = new AdamParameters(0.001, 0.9, 0.99, 0.00000001);

             network.Layers[0] = new Standard(batchSize, activator, new double[200], rempliMatrice(7,200), fixedL,1);
             network.Layers[1] = new Standard(batchSize, activator, new double[130], rempliMatrice(200,130), fixedL,2);
             network.Layers[2] = new Standard(batchSize, activator, new double[50], rempliMatrice( 130,50), fixedL,3);
             network.Layers[3] = new Standard(batchSize, activator, new double[1], rempliMatrice(50,1) , fixedL,4);
            var serialized = NetworkSerializer.Serialize(network);
            JsonSerializer serializer = new JsonSerializer();
            var filename = "../../../JSON/my-network_Standard_Sigmoid_FixedL.json"; //"../../../JSON/my-network_Sigmoid_Adam.json";  "../../../my-network_Sigmoid_Momemtum.json";
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }

            network.Layers[0] = new L2Penalty(new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), fixedL, 1), 0.1);
              network.Layers[1] = new L2Penalty(new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), fixedL, 2),  0.1);
              network.Layers[2] = new L2Penalty( new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), fixedL, 3),  0.1);
              network.Layers[3] = new L2Penalty( new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), fixedL, 4),0.1);

            serialized = NetworkSerializer.Serialize(network);
            serializer = new JsonSerializer();
            filename = "../../../JSON/my-network_Penalty_Sigmoid_Fixed.json"; 
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }
            network.Layers[0] = new WeightDecay(new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), fixedL, 1), 0.1);
            network.Layers[1] = new WeightDecay(new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), fixedL, 2), 0.1);
            network.Layers[2] = new WeightDecay(new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), fixedL, 3), 0.1);
            network.Layers[3] = new WeightDecay(new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), fixedL, 4), 0.1);

            serialized = NetworkSerializer.Serialize(network);
            serializer = new JsonSerializer();
            filename = "../../../JSON/my-network_WeightDecay_Sigmoid_Fixed.json";
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }



            network.Layers[0] = new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), adam, 1);
              network.Layers[1] = new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), adam, 2);
              network.Layers[2] = new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), adam, 3);
              network.Layers[3] = new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), adam, 4);

            serialized = NetworkSerializer.Serialize(network);
            serializer = new JsonSerializer();
            filename = "../../../JSON/my-network_Standard_Sigmoid_Adam.json";
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }

            network.Layers[0] = new L2Penalty(new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), adam, 1), 0.1);
             network.Layers[1] = new L2Penalty(new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), adam, 2),   0.1);
             network.Layers[2] = new L2Penalty( new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), adam, 3),  0.1);
             network.Layers[3] = new L2Penalty( new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), adam, 4), 0.1);

            serialized = NetworkSerializer.Serialize(network);
            serializer = new JsonSerializer();
            filename = "../../../JSON/my-network_Penalty_Sigmoid_Adam.json";
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }

            network.Layers[0] = new WeightDecay(new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), adam, 1), 0.1);
            network.Layers[1] = new WeightDecay(new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), adam, 2), 0.1);
            network.Layers[2] = new WeightDecay(new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), adam, 3), 0.1);
            network.Layers[3] = new WeightDecay(new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), adam, 4), 0.1);

            serialized = NetworkSerializer.Serialize(network);
            serializer = new JsonSerializer();
            filename = "../../../JSON/my-network_WeightDecay_Sigmoid_Adam.json";
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }



            network.Layers[0] = new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), mom, 1);
               network.Layers[1] = new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), mom, 2);
               network.Layers[2] = new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), mom, 3);
               network.Layers[3] = new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), mom, 4);
            serialized = NetworkSerializer.Serialize(network);
            serializer = new JsonSerializer();
            filename = "../../../JSON/my-network_Standard_Sigmoid_Mom.json";
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }

            network.Layers[0] = new L2Penalty(new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), mom, 1),  0.1);
            network.Layers[1] = new L2Penalty(new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), mom, 2), 0.1);
            network.Layers[2] = new L2Penalty(new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), mom, 3), 0.1);
            network.Layers[3] = new L2Penalty(new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), mom, 4), 0.1);
            serialized = NetworkSerializer.Serialize(network);
            serializer = new JsonSerializer();
            filename = "../../../JSON/my-network_Penalty_Sigmoid_Mom.json";
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }



            network.Layers[0] = new WeightDecay(new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), mom, 1), 0.1);
            network.Layers[1] = new WeightDecay(new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), mom, 2), 0.1);
            network.Layers[2] = new WeightDecay(new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), mom, 3), 0.1);
            network.Layers[3] = new WeightDecay(new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), mom, 4), 0.1);

            serialized = NetworkSerializer.Serialize(network);
            serializer = new JsonSerializer();
            filename = "../../../JSON/my-network_WeightDecay_Sigmoid_mom.json";
            using (StreamWriter sw = new StreamWriter(filename))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, serialized);
            }

            Console.Read();
        }
    }
}
