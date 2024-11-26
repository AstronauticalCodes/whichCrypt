using System;
using System.IO;
using System.Text.Json;
using System.Windows.Forms;
using System.Security.Cryptography;

namespace Demo1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            //InitializeCustomButton();
        }

        //string jsonFilePath = "PYtoCS.json";
        //string hashFilePath = "jsonHash.txt";

        //string currentHash = CalculateMD5Hash(jsonFilePath);

        //string storedHash = File.ReadAllText(hashFilePath);

        //if (currentHash != storedHash) {}

        //private void InitializeCustomButton()
        //{
        //    ButtonTrial customButton = new ButtonTrial();
        //    customButton.Text = "Click Me";
        //    customButton.Location = new System.Drawing.Point(50, 50);
        //    customButton.Click += CustomButton_Click;
        //    this.Controls.Add(customButton);
        //}
        //private void CustomButton_Click(object sender, EventArgs e)
        //{
        //    MessageBox.Show("Custom Button Clicked!");
        //}
        public void buttonTrial1_Click(object sender, EventArgs e)
        {
            string user = textBox1.Text;

            // Create and serialize the Person object
            var person = new Person { CipherText = user };
            string jsonString = JsonSerializer.Serialize(person);
            File.WriteAllText("CStoPY.json", jsonString);

            // Read and deserialize the JSON from PYtoCS.json
            try
            {
                string jsonString1 = File.ReadAllText("PYtoCS2.json");
                MessageBox.Show($"Read JSON: {jsonString1}");
                Person PYtoCS = JsonSerializer.Deserialize<Person>(jsonString1);
                MessageBox.Show($"CipherText: {PYtoCS.CipherText} , {PYtoCS.CipherText1}");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error reading or deserializing JSON: {ex.Message}");
            }
        }


        public class JsonHash
        {
            
            public string CipherText { get; set; }
        }

        public class JsonModel
        {
            public string Model { get; set; }
        }

        public class JsonPrediction
        {
            public Dictionary<string,Dictionary<string,string>> Data { get; set; }
        }

        var dataModel = JsonConvert.DeserializeObject<JsonPrediction>(jsonString);

        foreach (var key in dataModel.Data.Keys)
        {
            Console.WriteLine($"Key: {key}");
            foreach (var value in dataModel.Data[key])
            {
                Console.WriteLine($"  Value: {value}");
            }
        }

        //private void buttonTrial1_Click(object sender, EventArgs e)
        //{
        //    MessageBox.Show("Clicked");
        //}

        //static string CalculateMD5Hash(string filePath)
        //{
        //    using (var md5 = MD5.Create())
        //    using (var stream = File.OpenRead(filePath))
        //    {
        //        return BitConverter.ToString(md5.ComputeHash(stream)).Replace("-", string.Empty);
        //    }
        //}
    }
    class CStoPy
    {
        static void Hello()
        {
            string jsonFilePath = "PYtoCS2.json";
            string hashFilePath = "json_hash.txt";

            // Calculate hash
            string currentHash = CalculateMD5Hash(jsonFilePath);

            // Read stored hash
            string storedHash = File.ReadAllText(hashFilePath);

            // Compare hashes
            if (currentHash != storedHash)
            {
                //Console.WriteLine("JSON file has changed.");

                // Update stored hash
                File.WriteAllText(hashFilePath, currentHash);
            }
            else
            {
                Console.WriteLine("JSON file has not changed.");
            }
        }

        static string CalculateMD5Hash(string filePath)
        {
            using (var md5 = MD5.Create())
            using (var stream = File.OpenRead(filePath))
            {
                return BitConverter.ToString(md5.ComputeHash(stream)).Replace("-", string.Empty);
            }
        }
    }
}