import React, { useState } from 'react';
import {
  ChakraProvider,
  Container,
  Heading,
  Wrap,
  Input,
  Button,
  SkeletonCircle,
  SkeletonText,
  Stack,
  extendTheme,
} from '@chakra-ui/react';

const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);


  const customTheme = extendTheme({
    styles: {
      global: {
        body: {
          bg: 'gray.800', // Tutaj ustawiamy kolor tła na ciemnoszary, możesz dostosować odcień według własnych preferencji
          color: 'white', // Kolor tekstu
        },
      },
    },
  });



  const handleImageUpload = async () => {
    setLoading(true);

    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/process_image', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      const imageData = data.image_data;

      // Dekodowanie obrazu z Base64 i ustawienie wynikowego obrazu
      setImage(`data:image/png;base64, ${imageData}`);
    } catch (error) {
      console.error('Błąd:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  return (
    <ChakraProvider theme={customTheme}>
      <Container>
        <Heading marginTop={'20px'}>Single Pixel Camera 📷</Heading>
        <Wrap marginBottom={'20px'}>
          <Input marginTop={'20px'} type="file" accept=".png" onChange={handleFileChange}></Input>
          <Button onClick={handleImageUpload} colorScheme={'orange'}>
            Generuj
          </Button>
        </Wrap>

        {loading ? (
          <Stack>
            <SkeletonCircle />
            <SkeletonText />
          </Stack>
        ) : image ? (
          <img src={image} alt="Processed Image" style={{ boxShadow: 'lg' }} />
        ) : null}
      </Container>
    </ChakraProvider>
  );
};

export default ImageUpload;
