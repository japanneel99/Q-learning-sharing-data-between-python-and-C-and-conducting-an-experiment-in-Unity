using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Feeling : MonoBehaviour
{

    public float did_You_Feel_Safe;

    [SerializeField]
    public GameObject Pedestrian;


    // Start is called before the first frame update
    void Start()
    {
        did_You_Feel_Safe = 0.0f ;
    }

    // Update is called once per frame
    void Update()
    {

        if (Input.GetKeyDown(KeyCode.Space))
        {
            did_You_Feel_Safe = 1.0f;
            Debug.Log("The Space bar was pressed ");
        }
        else
        {
            did_You_Feel_Safe = -1.0f;
        }
    }

}