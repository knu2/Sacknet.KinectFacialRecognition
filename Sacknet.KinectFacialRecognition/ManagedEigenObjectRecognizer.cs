using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Knu2.Classifier;
using Emgu.CV;
using Microsoft.Kinect.Toolkit.FaceTracking;

namespace Sacknet.KinectFacialRecognition
{
    /// <summary>
    /// Based on the Emgu CV EigenObjectRecognizer, but converted to use fully managed objects.
    /// </summary>
    public class ManagedEigenObjectRecognizer
    {

        ERTreesClassifier forest;
        Dictionary<int, string> nameLookup;

        /// <summary>
        /// Initializes a new instance of the <see cref="ManagedEigenObjectRecognizer"/> class.
        /// </summary>
        public ManagedEigenObjectRecognizer(IEnumerable<TargetFace> targetFaces)
            : this(targetFaces, 2000)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ManagedEigenObjectRecognizer"/> class.
        /// </summary>
        public ManagedEigenObjectRecognizer(IEnumerable<TargetFace> targetFaces, double eigenDistanceThreshold)
            : this(targetFaces, eigenDistanceThreshold, targetFaces.Count(), 0.001)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ManagedEigenObjectRecognizer"/> class.
        /// </summary>
        public ManagedEigenObjectRecognizer(IEnumerable<TargetFace> targetFaces, double eigenDistanceThreshold, int maxIter, double eps)
        {
            Debug.Assert(eigenDistanceThreshold >= 0.0, "Eigen-distance threshold should always >= 0.0");

            Bitmap[] images = targetFaces.Select(x => x.Image).ToArray();
            FloatImage[] eigenImages;
            FloatImage averageImage;

            CalcEigenObjects(images, maxIter, eps, out eigenImages, out averageImage);

            this.EigenValues = images.Select(x => EigenDecomposite(x, eigenImages, averageImage)).ToArray();
            this.EigenImages = eigenImages;
            this.AverageImage = averageImage;
            this.Labels = targetFaces.Select(x => x.Key).ToArray();
            this.EigenDistanceThreshold = eigenDistanceThreshold;

            nameLookup = new Dictionary<int, string>();
            foreach (var face in targetFaces)
            {
                
                nameLookup[RecognitionUtility.Shorten(face.ID)] = face.Key;
            }

            try
            {
                forest = new ERTreesClassifier();
                forest.Train(targetFaces);
            }
            catch (NullReferenceException)
            {
                // will occur if there is no 3d face points
            }
        }

        /// <summary>
        /// Gets or sets the eigen vectors that form the eigen space
        /// </summary>
        /// <remarks>The set method is primary used for deserialization, do not attemps to set it unless you know what you are doing</remarks>
        public FloatImage[] EigenImages { get; set; }

        /// <summary>
        /// Gets or sets the labels for the corresponding training image
        /// </summary>
        public string[] Labels { get; set; }

        /// <summary>
        /// Gets or sets the eigen distance threshold.
        /// The smaller the number, the more likely an examined image will be treated as unrecognized object. 
        /// Set it to a huge number (e.g. 5000) and the recognizer will always treated the examined image as one of the known object. 
        /// </summary>
        public double EigenDistanceThreshold { get; set; }

        /// <summary>
        /// Gets or sets the average Image. 
        /// </summary>
        /// <remarks>The set method is primary used for deserialization, do not attemps to set it unless you know what you are doing</remarks>
        public FloatImage AverageImage { get; set; }

        /// <summary>
        /// Gets or sets the eigen values of each of the training image
        /// </summary>
        /// <remarks>The set method is primary used for deserialization, do not attemps to set it unless you know what you are doing</remarks>
        public float[][] EigenValues { get; set; }

        /// <summary>
        /// Caculate the eigen images for the specific traning image
        /// </summary>
        public static void CalcEigenObjects(Bitmap[] trainingImages, int maxIter, double eps, out FloatImage[] eigenImages, out FloatImage avg)
        {
            int width = trainingImages[0].Width;
            int height = trainingImages[0].Height;

            if (maxIter <= 0 || maxIter > trainingImages.Length)
                maxIter = trainingImages.Length;

            int maxEigenObjs = maxIter;

            eigenImages = new FloatImage[maxEigenObjs];
            for (int i = 0; i < eigenImages.Length; i++)
                eigenImages[i] = new FloatImage(width, height);

            avg = new FloatImage(width, height);

            ManagedEigenObjects.CalcEigenObjects(trainingImages, maxIter, eps, eigenImages, null, avg);
        }

        /// <summary>
        /// Decompose the image as eigen values, using the specific eigen vectors
        /// </summary>
        public static float[] EigenDecomposite(Bitmap src, FloatImage[] eigenImages, FloatImage avg)
        {
            return ManagedEigenObjects.EigenDecomposite(src, eigenImages, avg);
        }

        /// <summary>
        /// Get the Euclidean eigen-distance between <paramref name="image"/> and every other image in the database
        /// </summary>
        public float[] GetEigenDistances(Bitmap image)
        {
            var decomp = EigenDecomposite(image, this.EigenImages, this.AverageImage);

            List<float> result = new List<float>();

            foreach (var eigenValue in this.EigenValues)
            {
                // norm = ||arr1-arr2||_L2 = sqrt( sum_I (arr1(I)-arr2(I))^2 )
                double sum = 0;

                for (var i = 0; i < eigenValue.Length; i++)
                {
                    sum += Math.Pow(decomp[i] - eigenValue[i], 2);
                }

                result.Add((float)Math.Sqrt(sum));
            }

            return result.ToArray();
        }

        /// <summary>
        /// Given the <paramref name="image"/> to be examined, find in the database the most similar object, return the index and the eigen distance
        /// </summary>
        public void FindMostSimilarObject(Bitmap image, out int index, out float eigenDistance, out string label)
        {
            float[] dist = this.GetEigenDistances(image);

            index = 0;
            eigenDistance = dist[0];

            for (int i = 1; i < dist.Length; i++)
            {
                if (dist[i] < eigenDistance)
                {
                    index = i;
                    eigenDistance = dist[i];
                }
            }

            label = this.Labels[index];
        }

        /// <summary>
        /// Try to recognize the image and return its label
        /// </summary>
        public string Recognize(Bitmap image, out float eigenDistance)
        {
            int index;
            string label;
            this.FindMostSimilarObject(image, out index, out eigenDistance, out label);

            return (this.EigenDistanceThreshold <= 0 || eigenDistance < this.EigenDistanceThreshold) ? this.Labels[index] : string.Empty;
        }

        /// <summary>
        /// Try to recognize the given face 3D points
        /// </summary>
        /// <param name="face3DPoints">the collection of face 3D points</param>
        /// <returns>the name of the face</returns>
        public string Recognize(EnumIndexableCollection<FeaturePoint, Vector3DF> face3DPoints)
        {
            var id = forest.Recognize(face3DPoints);
            Debug.WriteLine("Recognized ID = " + id); 
            return nameLookup[RecognitionUtility.Shorten(id)];
        }
    }

    /// <summary>
    /// Contains extension methods to support the different classifier methods
    /// </summary>
    public static class ClassifierHelper
    {
        /// <summary>
        /// Prepares training data and calls ERTreesClassifier.Train method
        /// </summary>
        /// <param name="forest">ERTreesClassifier object</param>
        /// <param name="targetFaces">collection of targetFaces</param>
        static public void Train(this ERTreesClassifier forest, IEnumerable<TargetFace> targetFaces)
        {
            var first = targetFaces.ElementAt(0);
            var varCount = first.Face3DPoints.Count; // x, y, z 

            Matrix<float> data, responses;            
            data = new Matrix<float>(targetFaces.Count(), varCount*3);
            responses = new Matrix<float>(targetFaces.Count(), 1);

            int i = 0;

            foreach (var face in targetFaces)
            {
                for (int j = 0; j < varCount; j++)
                {
                    data[i, j * 3] = face.Face3DPoints[j].X;
                    data[i, j * 3 + 1] = face.Face3DPoints[j].Y;
                    data[i, j * 3 + 2] = face.Face3DPoints[j].Z;
                }

                responses[i, 0] = (float)face.ID;
                i++;
                
            }
            var numClasses = targetFaces.Distinct().Count();


            forest.Train(data, responses, numClasses);
        }

        /// <summary>
        /// Prepares the 3d face points and calls the ERTreesClassifier.Predict method
        /// </summary>
        /// <param name="forest">ERTreesClassifier object</param>
        /// <param name="face3DPoints">3d points of the face</param>
        /// <returns>ID of the face</returns>
        static public int Recognize(this ERTreesClassifier forest, EnumIndexableCollection<FeaturePoint, Vector3DF> face3DPoints)
        {           
            var varCount = face3DPoints.Count; // x, y, z 

            Matrix<float> data;
            data = new Matrix<float>(1, face3DPoints.Count()*3);
            
            for (int j = 0; j < varCount; j++)
            {
                data[0, j * 3] = face3DPoints[j].X;
                data[0, j * 3 + 1] = face3DPoints[j].Y;
                data[0, j * 3 + 2] = face3DPoints[j].Z;
            }

            var r = forest.Predict(data);
            return (int)r;
        }
    }
}