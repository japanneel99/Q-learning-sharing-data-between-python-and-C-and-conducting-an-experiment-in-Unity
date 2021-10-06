using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;

public class Connection : MonoBehaviour
{
    [SerializeField]
    public GameObject Pedestrian;

    Thread mThread;
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 25001;
    TcpListener listener;
    TcpClient client;
    Vector3 wheelchairVelocity;
    Vector3 wheelchairPosition;
    Vector3 pedestrianPosition;
    Vector3 pedestrianVelocity;
    Vector3 Initial_Wheelchair_Position;
    Vector3 Initial_Pedestrian_Position;
    Vector3 Initial_Wheelchair_Velocity;
    Vector3 Initial_Pedestrian_Velocity;
    public float feeling;
    IPAddress localAdd;
    bool running;
    bool do_reset;
    float received_action;


    // Update is called once per frame
    private void Update()
    {
        UpdateCurrentState();

        if (do_reset)
        {
            ResetState();
            do_reset = false;
        }
    }

    private void Start()
    {
        //これを追加してpythonのリセット関数で使う?　このコードのsendAndrecieve関数でpythonに送信する。
        Initial_Wheelchair_Position = transform.position;
        Initial_Pedestrian_Position = Pedestrian.transform.position;

        Initial_Wheelchair_Velocity = this.GetComponent<PMController>().CurrentVelocity;
        Initial_Pedestrian_Velocity = Pedestrian.GetComponent<NetworkCharacterVelocity>().velocity;
        do_reset = false;
        UpdateCurrentState();

        ThreadStart ts = new ThreadStart(GetInfo);
        mThread = new Thread(ts);
        mThread.Start();

        received_action = 0;
    }

    void UpdateCurrentState()
    {
        GetComponent<PMController>().velocity = received_action;

        wheelchairPosition = transform.position;
        wheelchairVelocity = this.GetComponent<PMController>().CurrentVelocity;

        pedestrianPosition = Pedestrian.transform.position;
        pedestrianVelocity = Pedestrian.GetComponent<NetworkCharacterVelocity>().velocity;

        feeling = GetComponent<Feeling>().did_You_Feel_Safe;
    }

    void ResetState()
    {
        transform.position = Initial_Wheelchair_Position;
        Pedestrian.transform.position = Initial_Pedestrian_Position;

        Initial_Wheelchair_Velocity = this.GetComponent<PMController>().CurrentVelocity;
        Pedestrian.GetComponent<NetworkCharacterVelocity>().velocity = Initial_Pedestrian_Velocity;

        UpdateCurrentState();
    }

    void GetInfo()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(IPAddress.Any, connectionPort);
        listener.Start();

        client = listener.AcceptTcpClient();

        running = true;
        while (running)
        {
            SendAndReceiveData();
        }
        listener.Stop();
    }

    void SendAndReceiveData()
    {
        NetworkStream nwStream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize];

        //----receiving data from host-----//
        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize); //Getting data in bytes from python. パイソンからバイトでデータを取得
        string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead); // Converting byte data to string. バイトデータをストリングに変換

        if (dataReceived != null)
        {
            Debug.Log(string.Format("received data:{0}", dataReceived));
            var xs = dataReceived.Split(',');
            var cmd = xs[0];
            var send_message = "";

            switch (cmd)
            {
                //    case "GET_INITIAL_WHEELCHAIR_POSITION":
                //        send_message = JsonUtility.ToJson(Initial_Wheelchair_Position);
                //        break;
                //    case "GET_INITIAL_PEDESTRIAN_POSITION":
                //        send_message = JsonUtility.ToJson(Initial_Pedestrian_Position);
                //        break;
                //    case "GET_INITIAL_WHEELCHAIR_VELOCITY":
                //        send_message = JsonUtility.ToJson(Initial_Wheelchair_Velocity);
                //        break;
                //    case "GET_INITIAL_PEDESTRIAN_VELOCITY":
                //        send_message = JsonUtility.ToJson(Initial_Pedestrian_Velocity);
                //        break;
                case "GET_WHEELCHAIR_POSITION":
                    send_message = JsonUtility.ToJson(wheelchairPosition);
                    break;
                case "GET_WHEELCHAIR_VELOCITY":
                    send_message = JsonUtility.ToJson(wheelchairVelocity);
                    break;
                case "GET_PEDESTRIAN_POSITION":
                    send_message = JsonUtility.ToJson(pedestrianPosition);
                    break;
                case "GET_PEDESTRIAN_VELOCITY":
                    send_message = JsonUtility.ToJson(pedestrianVelocity);
                    break;
                case "GET_FEELING_REWARD":
                    send_message = string.Format("{{\"info\": {0}}}", feeling);
                    break;
                case "SEND_ACTION":
                    send_message = "received action without any issues";
                    received_action = float.Parse(xs[1]);
                    Debug.Log(string.Format("parsed data: {0}", received_action));
                    break;
                case "RESET":
                    do_reset = true;
                    while(do_reset)
                    {
                        Debug.Log("reset executing");
                        sleep(0.1f);
                    }
                    Debug.Log(string.Format("reset"));
                    send_message = "reset_executed";
                    break;
                default:
                    send_message = "unknown command";
                    break;
            }
            Debug.Log(string.Format("return msg: {0}", send_message));
            //---Sending data to python... パイソンにデータを送信
            //---Converting string to byte data ストリングをバイトに変換
            byte[] myWriteBuffer = Encoding.ASCII.GetBytes(send_message);
            nwStream.Write(myWriteBuffer, 0, myWriteBuffer.Length); //Sending to python.パイソンに送信
        }

    }
    IEnumerator sleep(float t)
    {
        yield return new WaitForSeconds(t);

    }

    public static Vector3 StringToVector3(string sVector)
    {
        //remove the parentheses
        if (sVector.StartsWith("(") && sVector.EndsWith(")"))
        {
            sVector = sVector.Substring(1, sVector.Length - 2);
        }

        //split the items, 要素を分解する
        string[] sArray = sVector.Split(',');

        //store as a vector3 Vector3として保存する
        Vector3 result = new Vector3(
            float.Parse(sArray[0]),
            float.Parse(sArray[1]),
            float.Parse(sArray[2]));

        return result;
    }

}