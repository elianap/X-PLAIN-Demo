import React, { useEffect, useState } from "react"
import { Redirect } from "react-router-dom"
import Container from "react-bootstrap/Container"
import Carousel from 'react-bootstrap/Carousel'

function Home() {
  const [toClassifiers, setToClassfiers] = useState(false)


  if (toClassifiers) {
    return <Redirect to="/classifiers" />
  }
  return (
    <Container>
<Carousel>
  <Carousel.Item>
    <img
      className="d-block w-100"
      src="holder.js/800x400?text=First slide&bg=373940"
      alt="First slide"
    />
    <Carousel.Caption>
      <h3>First slide label</h3>
      <p>Nulla vitae elit libero, a pharetra augue mollis interdum.</p>
    </Carousel.Caption>
  </Carousel.Item>
</Carousel>
    </Container>
  )
}

export default Home
