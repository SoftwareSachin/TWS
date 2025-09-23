import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Connect Your Data Sources',
    description: (
      <>
        Amplifi seamlessly connects to your existing data sources including databases, 
        cloud storage, document libraries, and APIs - bringing all your enterprise 
        knowledge into one searchable platform.
      </>
    ),
  },
  {
    title: 'AI-Powered Intelligence',
//    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Ask questions in natural language and get accurate answers based on your own data.
        Amplifi uses advanced AI to understand context, extract insights, and 
        make information discoverable.
      </>
    ),
  },
  {
    title: 'Enterprise-Ready Solutions',
//    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Deploy powerful solutions for knowledge management, customer support automation, 
        and market intelligence. Amplifi is secure, scalable, and designed for 
        business-critical applications.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
