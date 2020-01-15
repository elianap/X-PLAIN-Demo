import "./style.scss"

import "whatwg-fetch"

import React, { useState } from "react"

import { Link, Redirect, Route, Switch, useLocation } from "react-router-dom"

import Navbar from "react-bootstrap/Navbar"
import Nav from "react-bootstrap/Nav"
import NavItem from "react-bootstrap/NavItem"

import Octicon, { Book, Graph, Italic, Telescope } from "@primer/octicons-react"

import Datasets from "./Datasets"
import Classifiers from "./Classifiers"
import Instances from "./Instances"
import Analyses from "./Analyses"

import Explanation from "./Explanation"
import WhatIf from "./WhatIf"

import MispredInstances from "./MispredInstances"
import InstancesExplClassCompare from "./InstancesExplClassCompare"

import UserRules from "./UserRules"
import ExplanationComparison from "./ExplanationComparison"

import GlobalExplanation from "./GlobalExplanation"

import ShowInstances from "./ShowInstances"
import Classifiers2 from "./Classifiers2"
import AnalysesNew from "./AnalysesNew"

import ExplanationClassComparison from "./ExplanationClassComparison"

import ClassCompare from "./ClassCompare"

function RouteNotFound() {
  return <h1>Route not found</h1>
}

function App() {
  const location = useLocation()
  const [instance, setInstance] = useState(null)

  return (
    <Route path="/">
      <main>
        <Navbar bg="dark" variant="dark" expland="lg">
          <Navbar.Brand as={Link} to="/">
            X-PLAIN
          </Navbar.Brand>
          <Navbar.Collapse>
            <Nav
              activeKey={location.pathname}
              navbar={true}
              className="mr-auto"
            >
              <NavItem href="/datasets">
                <Nav.Link as={Link} eventKey="/datasets" to="/datasets">
                  <Octicon icon={Book} /> Datasets
                </Nav.Link>
              </NavItem>
              <NavItem href="/classifiers">
                <Nav.Link as={Link} eventKey="/classifiers" to="/classifiers">
                  <Octicon icon={Telescope} /> Classifiers
                </Nav.Link>
              </NavItem>
              <NavItem href="/show_instances">
                <Nav.Link
                  as={Link}
                  eventKey="/show_instances"
                  to="/show_instances"
                >
                  <Octicon icon={Italic} /> Show Instances
                </Nav.Link>
              </NavItem>
              <NavItem href="/analyses">
                <Nav.Link as={Link} eventKey="/analyses" to="/analyses">
                  <Octicon icon={Graph} /> Analyses
                </Nav.Link>
              </NavItem>
            </Nav>
            {instance !== null ? (
              <Navbar.Text className="mr-sm-2">
                Current instance: <strong>{instance}</strong>
              </Navbar.Text>
            ) : null}
          </Navbar.Collapse>
        </Navbar>

        <Switch>
          <Route path="/datasets">
            <Datasets />
          </Route>

          <Route path="/classifiers">
            <Classifiers />
          </Route>

          <Route path="/instances_class_comparison">
            <InstancesExplClassCompare />
          </Route>

          <Route path="/instances">
            <Instances setInstance={setInstance} />
          </Route>

          <Route path="/mispred_instances">
            <MispredInstances />
          </Route>

          <Route path="/analyses">
            <Analyses />
          </Route>

          <Route path="/whatif">
            <WhatIf />
          </Route>

          <Route path="/explanation">
            <Explanation />
          </Route>

          <Route path="/show_instances">
            <ShowInstances />
          </Route>

          <Route path="/explanation_comparison">
            <ExplanationComparison />
          </Route>

          <Route path="/explanation_class_comparison">
            <ExplanationClassComparison />
          </Route>
          <Route path="/class_comparison">
            <ClassCompare />
          </Route>

          <Route path="/global_explanation">
            <GlobalExplanation />
          </Route>

          <Route path="/user_rules">
            <UserRules />
          </Route>

          <Route exact path="/">
            <Redirect to="/datasets" />
          </Route>

          <Route path="/classifiers_2">
            <Classifiers2 />
          </Route>

          <Route path="/analyses_new">
            <AnalysesNew />
          </Route>

          <Route component={RouteNotFound} />
        </Switch>
      </main>
    </Route>
  )
}

export default App
