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
            IActivator activator = new Tanh(); // ReLU();
            int numberLayers = 4;
            Network network = new Network(batchSize, numberLayers);

            double[,] weights = new double[7, 200];

            /*network.Layers[0] = new Standard(batchSize, activator, new double[200], rempliMatrice(7,200), new FixedLearningRateParameters(0.1),1);
            network.Layers[1] = new Standard(batchSize, activator, new double[130], rempliMatrice(200,130), new FixedLearningRateParameters(0.1),2);
            network.Layers[2] = new Standard(batchSize, activator, new double[50], rempliMatrice( 130,50), new FixedLearningRateParameters(0.1),3);
            network.Layers[3] = new Standard(batchSize, activator, new double[1], rempliMatrice(50,1) , new FixedLearningRateParameters(0.1),4);*/

           /* network.Layers[0] = new L2Penalty(new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), new FixedLearningRateParameters(0.1), 1), 1, 0.1);
            network.Layers[1] = new L2Penalty(new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), new FixedLearningRateParameters(0.1), 2), 2,  0.1);
            network.Layers[2] = new L2Penalty( new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), new FixedLearningRateParameters(0.1), 3), 3, 0.1);
            network.Layers[3] = new L2Penalty( new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), new FixedLearningRateParameters(0.1), 4), 4,0.1);*/

            /* network.Layers[0] = new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), new AdamParameters(0.01, 0.9, 0.99,0.0000001), 1);
             network.Layers[1] = new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), new AdamParameters(0.01, 0.9, 0.99, 0.0000001), 2);
             network.Layers[2] = new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), new AdamParameters(0.01, 0.9, 0.99, 0.0000001), 3);
             network.Layers[3] = new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), new AdamParameters(0.01, 0.9, 0.99, 0.0000001), 4);

             network.Layers[0] = new L2Penalty(new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), new AdamParameters(0.01, 0.9, 0.99, 0.0000001), 1), 1, 0.1);
            network.Layers[1] = new L2Penalty(new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), new AdamParameters(0.01, 0.9, 0.99,0.0000001), 2), 2,  0.1);
            network.Layers[2] = new L2Penalty( new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), new AdamParameters(0.01, 0.9, 0.99,0.0000001), 3), 3, 0.1);
            network.Layers[3] = new L2Penalty( new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), new AdamParameters(0.01, 0.9, 0.99,0.0000001), 4), 4,0.1);*/


            network.Layers[0] = new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), new MomentumParameters(), 1);
              network.Layers[1] = new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), new MomentumParameters(), 2);
              network.Layers[2] = new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), new MomentumParameters(), 3);
              network.Layers[3] = new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), new MomentumParameters(), 4);


            network.Layers[0] = new L2Penalty(new Standard(batchSize, activator, new double[200], rempliMatrice(7, 200), new MomentumParameters(), 1),  0.1);
            network.Layers[1] = new L2Penalty(new Standard(batchSize, activator, new double[130], rempliMatrice(200, 130), new MomentumParameters(), 2), 0.1);
            network.Layers[2] = new L2Penalty(new Standard(batchSize, activator, new double[50], rempliMatrice(130, 50), new MomentumParameters(), 3), 0.1);
            network.Layers[3] = new L2Penalty(new Standard(batchSize, activator, new double[1], rempliMatrice(50, 1), new MomentumParameters(), 4), 0.1);
            var serialized = NetworkSerializer.Serialize(network);
             JsonSerializer serializer = new JsonSerializer();
             var filename = "../../../JSON/my-network_Penalty_Tanh_Momemtum.json"; //"../../../JSON/my-network_Tanh_Adam.json";  "../../../my-network_Tanh_Momemtum.json";
            using (StreamWriter sw = new StreamWriter(filename))
             using (JsonWriter writer = new JsonTextWriter(sw))
             {
                 serializer.Serialize(writer, serialized);
             }
            Console.Read();
        }
    }
}
