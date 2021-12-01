// There is only one class called App
// the constructor() does initialzation job
// handleChange() is triggered when input changes
// handleClick() is triggered when clicking any buttons
// render() inlude the page desgin
// you may want to start read render() first
import React from 'react';
import { 
    Row, Col, Button,
    Card,  CardBody, CardHeader, CardFooter,
    Input, InputGroup, InputGroupAddon, Spinner,
    } 
    from 'reactstrap';
import './App.css';

class App extends React.Component {
  // ######################### initialzation ###########################
  constructor(props) {
    super(props);
    this.state = {
        "tweetUrl": "",
        "clipRes": "",
        "mfasRes": "",
        "httpStatus": 0,
        "resMsg": "",
        "loading": false
        };
    this.handleClick = this.handleClick.bind(this);
    this.handleChange = this.handleChange.bind(this);
  }

  // ######################### input handler ###########################
  handleChange(event) {
    const inputId = event.target.id;
    const inputValue = event.target.value;
    let state = this.state;
    state[inputId] = inputValue;
    this.setState(state);
  }

  // ######################### click handler ###########################
  handleClick(event) {
    const serverUrl = "http://localhost:5000/predict";
    // const debugUrl = "http://localhost:5000/debug";
    const buttonId = event.target.id;
    let state = this.state
    // ######################### submit tweet url ###########################
    if (buttonId === "submit") {
      state["loading"] = true;
      this.setState(state);
      fetch(serverUrl, {
        method: 'POST',
        headers: {
          'Accept': 'application/json; charset=UTF-8',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            tweetUrl: state["tweetUrl"],
            })
      })
      .then(response => {
        // console.log(response);
        state["httpStatus"] = response.status;
        state["resMsg"] = response.statusText;
        state["loading"] = false;
        if (response.status < 300) {
          return response.json();
        }
        return {}
      })
      .then(object => {
          // console.log(object);
          state["loading"] = false;
          if (state["httpStatus"] < 300) {
            state["clipRes"] = object["clip"];
            state["mfasRes"] = object["mfas"];
          }
          this.setState(state);
      })
      .catch(error => {
        // console.log(error);
        state["loading"] = false;
        state["resMsg"] = error;
        this.setState(state);
      });    
    }
  }

  // ######################### page design ###########################
  render() {
    const state = this.state;
    return (
    <Row> <Col sm={{ size: 6, offset: 3 }}> 
    <Card className='mt-5'>
      <CardHeader tag="h3">Vaccine Tweet Checker</CardHeader>
      <CardBody>
        <InputGroup>
          <InputGroupAddon addonType="prepend">Tweet Url:</InputGroupAddon>
          <Input value={state["tweetUrl"]} onChange={this.handleChange} id="tweetUrl"/>
        </InputGroup> <br />
        <Button color="success" onClick={this.handleClick} id="submit">Submit</Button>{" "}
      </CardBody>
      {state["loading"] ?         
        <div>
          <CardFooter>
                <Spinner
                as="span"
                variant="light"
                size="sm"
                role="status"
                aria-hidden="true"
                animation="border"/>
                  {"  Loading..."}

            </CardFooter>
        </div> 
        :
        <CardFooter> 
          {"Message: " + state["resMsg"]} <br />
          {"MFAS Result: " + state["mfasRes"]} <br />
          {"CLIP Result: " + state["clipRes"]} <br />
        </CardFooter>
      }
    </Card> </Col> </Row>
    )
  }
}

export default App;
